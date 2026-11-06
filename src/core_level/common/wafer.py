import os
import logging
import numpy as np
import pandas as pd

from typing import List
from src.core_level.common.isa import InstructionSet


def write_to_csv(dat, fname):
    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))

    with open(fname, "w") as f:
        for line in dat:
            f.write(line)
            f.write("\n")

class Core:
    def __init__(self, node_id: int, local_id: int, num_cores_per_node: int) -> None:
        self.node_id = node_id
        self.local_id = local_id
        self.core_id = node_id * num_cores_per_node + local_id
        self.instruction_queue = []
        self.traces = []
        self.next_id = 0

    def __str__(self):
        return str(self.core_id)

    def add_instruction(self, tile_op: "TileGemmOp"):
        # NOTE: the tile op itself is deliberately NOT stored here (it used to be appended to
        # self.instruction_queue, which nothing ever read). Keeping it pinned every op together
        # with its tiles/stats for the whole run, i.e. memory grew with the number of layers.
        # Each tile-op emits Instructions with local ids contiguous from 0.
        # Rebase them to this core's file-wide id space on append, shifting
        # both local_id and local_deps by the same constant offset.
        base = self.next_id
        for instr in tile_op.get_traces():
            gid = base + instr.local_id
            gdeps = [base + d for d in instr.local_deps]
            self.traces.append(instr.render(gid, gdeps))
            self.next_id = gid + 1
        logging.debug("TileGemmOp {} is added to core {} instruction queue.".format(tile_op.id, self.core_id))
    
    def get_traces(self):
        return self.traces

class MemoryBank:
    def __init__(self, node_id: int, local_id: int, num_banks_per_node: int) -> None:
        self.node_id = node_id
        self.local_id = local_id
        self.bank_id = node_id * num_banks_per_node + local_id
        self.allocated_tiles = []

    def alloc_tile(self, tile: "Tile"):
        self.allocated_tiles.append(tile)
        logging.debug("Tile {} is allocated to memory bank {}.".format(tile.id, self.bank_id))

class Wafer:
    def __init__(self, node_grid, core_grid) -> None:
        assert len(node_grid) == 2, "node_grid should be a tuple of (rows, columns)"
        self.node_grid = node_grid  # (rows, columns)
        self.num_nodes = node_grid[0] * node_grid[1]

        self.core_grid = core_grid  # (rows, columns)
        self.num_cores_per_node = core_grid[0] * core_grid[1]
        self.num_banks_per_node= core_grid[0] * core_grid[1]  # assuming same grid for cores and memory banks

        self.banks = {}
        self.cores = {}
        for node_id in range(self.num_nodes):
            self.banks[node_id] = {}
            self.cores[node_id] = {}
            for i in range(self.num_banks_per_node):
                self.banks[node_id][i] = MemoryBank(node_id, i, self.num_banks_per_node)

            for i in range(self.num_cores_per_node):
                self.cores[node_id][i] = Core(node_id, i, self.num_cores_per_node)

    def get_bank(self, node_id: int, local_id: int) -> MemoryBank:
        return self.banks[node_id][local_id]
    
    def get_core(self, node_id: int, local_id: int) -> Core:
        return self.cores[node_id][local_id]
    
    def get_core_by_global_id(self, global_id: int) -> Core:
        node_id = global_id // self.num_cores_per_node
        local_id = global_id % self.num_cores_per_node
        return self.cores[node_id][local_id]

    def local_id_to_global(self, node_id: int, local_id: int) -> int:
        return node_id * self.num_cores_per_node + local_id

    def get_traces(self):
        traces = {}
        for node_id in range(self.num_nodes):
            traces[node_id] = {}
            for core_id in range(self.num_cores_per_node):
                traces[node_id][core_id] = self.cores[node_id][core_id].get_traces()
        return traces

    def _trace_fname(self, iter, dir_path, node_id, core_id):
        return dir_path + "/node_{}/{}/core_{}.csv".format(node_id, iter, core_id)

    def flush_traces(self, iter, dir_path: str):
        """Move the trace lines every core has collected so far to its file and free them.

        The first write to a file in this Wafer's lifetime truncates it (exactly what export_traces
        did), later writes append, so the final files are identical to a single export at the end.
        Call it every now and then during generation to keep memory flat instead of holding every
        rendered instruction of the whole run until export_traces().
        After a flush, Core.get_traces()/Wafer.get_traces() only return the lines not written yet.
        """
        debug = logging.getLogger().isEnabledFor(logging.DEBUG)
        for node_id in range(self.num_nodes):
            for core_id in range(self.num_cores_per_node):
                core = self.cores[node_id][core_id]
                if not core.traces:
                    continue
                if debug:  # same debug lines export_traces used to print (in flush order, not file order)
                    for trace in core.traces:
                        logging.debug("Core {}: {}".format(core_id, trace))
                self._write_lines(self._trace_fname(iter, dir_path, node_id, core_id), core.traces)
                core.traces = []

    def _write_lines(self, fname, lines):
        started = getattr(self, "_flushed_files", None)
        if started is None:
            started = self._flushed_files = set()
        if fname not in started:
            folder = os.path.dirname(fname)
            if not os.path.exists(folder):
                os.makedirs(folder)
            mode = "w"
            started.add(fname)
        else:
            mode = "a"
        with open(fname, mode) as f:
            f.writelines(line + "\n" for line in lines)

    def export_traces(self, iter:int, dir_path: str):
        debug = logging.getLogger().isEnabledFor(logging.DEBUG)
        for node_id in range(self.num_nodes):
            for core_id in range(self.num_cores_per_node):
                core_traces = self.cores[node_id][core_id].get_traces()

                if debug:
                    for trace in core_traces:
                        logging.debug("Core {}: {}".format(core_id, trace))

                # every core gets a file, even one with no instructions (an empty file)
                self._write_lines(self._trace_fname(iter, dir_path, node_id, core_id), core_traces)

        # A later export overwrites the files again with the lines still held in memory. Without
        # flush_traces() that is everything (the original behaviour); after flushes it is only the
        # lines added since the last flush, so call export_traces() once, at the end, when flushing.
        self._flushed_files = set()

    def load_traces(self, iter: int, dir_path: str, filter_by_uid: str = None):
        traces = {}
        for node_id in range(self.num_nodes):
            traces[node_id] = {}
            for core_id in range(self.num_cores_per_node):
                in_fname = dir_path + "/node_{}/{}/core_{}.csv".format(node_id, iter, core_id)
                if not os.path.exists(in_fname):
                    logging.warning("Trace file {} does not exist.".format(in_fname))
                    continue

                with open(in_fname, "r") as f:
                    trace = f.readlines()
                    trace = [line.strip() for line in trace]

                    if filter_by_uid:
                        traces[node_id][core_id] = [line for line in trace if filter_by_uid in line.split(";")[-1]]
                    else:
                        traces[node_id][core_id] = trace
        return traces
        
    def extract_traffic(self, traces):
        traffic = []
        for node_id in range(self.num_nodes):
            for core_id in range(self.num_cores_per_node):
                for trace in traces[node_id][core_id]:
                    parsed_instr = InstructionSet.parse(trace)
                    print(f"{core_id}:", parsed_instr)
                    if parsed_instr[0] == "READ":
                        # parsed core/bank ids are global ids, convert to local
                        src_node = parsed_instr[1] // self.num_cores_per_node
                        src_bank = parsed_instr[1] % self.num_cores_per_node
                        dst_node = node_id
                        dst_core = core_id
                        size = parsed_instr[2]
                        # self.register_bank_to_core(src_node, src_bank, node_id, core_id, size)
                        traffic.append({"type": "READ", "src": f"{src_node}:B{src_bank}", "dst": f"{dst_node}:C{dst_core}", "size": size})
                        logging.debug(f"READ from {src_node}:{src_bank} size {size}")
                    elif parsed_instr[0] == "WRITE":
                        src_node = node_id
                        src_core = core_id
                        dst_node = parsed_instr[1] // self.num_cores_per_node
                        dst_bank = parsed_instr[1] % self.num_cores_per_node
                        size = parsed_instr[2] 
                        # self.register_core_to_bank(node_id, core_id, dst_node, dst_bank, size)
                        traffic.append({"type": "WRITE", "src": f"{src_node}:C{src_core}", "dst": f"{dst_node}:B{dst_bank}", "size": size})
                        logging.debug(f"WRITE to {dst_node}:{dst_bank} size {size}")
                    elif parsed_instr[0] == "COPY":
                        src_node = parsed_instr[1] // self.num_banks_per_node
                        assert src_node == node_id, "Source node must be the current node"
                        src_bank = parsed_instr[1] % self.num_banks_per_node
                        dst_node = parsed_instr[2] // self.num_banks_per_node
                        dst_bank = parsed_instr[2] % self.num_banks_per_node
                        size = parsed_instr[3]
                        # self.register_bank_to_bank(src_node, src_bank, dst_node, dst_bank, size)
                        # traffic.append((src_node, f"B:{src_bank}", dst_node, f"B:{dst_bank}", size))
                        traffic.append({"type": "COPY", "src": f"{src_node}:B{src_bank}", "dst": f"{dst_node}:B{dst_bank}", "size": size})
                        logging.debug(f"COPY from {src_node}:{src_bank} to {dst_node}:{dst_bank} size {size}")
                    elif parsed_instr[0] == "MULTICAST":
                        src_node = parsed_instr[1] // self.num_banks_per_node
                        assert src_node == node_id, "Source node must be the current node"
                        src_bank = parsed_instr[1] % self.num_banks_per_node
                        
                        dsts = []
                        for dst in parsed_instr[2]:
                            dst_node = dst // self.num_banks_per_node
                            dst_bank = dst % self.num_banks_per_node
                            dsts.append((dst_node, dst_bank))
                            size = parsed_instr[3]
                            # self.register_bank_to_bank(src_node, src_bank, dst_node, dst_bank, size)
                        
                        dsts = ",".join([f"{dst_node}:B{dst_bank}" for dst_node, dst_bank in dsts])
                        # traffic.append((src_node, f"B:{src_bank}", dst_node, f"B:{dst_bank}", size))
                        traffic.append({"type": "MULTICAST", "src": f"{src_node}:B{src_bank}", "dst": dsts, "size": size})

                        logging.debug(f"MULTICAST from {src_node}:{src_bank} to {dsts} size {size}")
                    else:
                        pass

        return traffic

    def calc_comm_matrix(self, traffic, fname):
        assert self.num_banks_per_node == self.num_cores_per_node, "This function assumes the same number of cores and banks per node."
        units = []
        for node_id in range(self.num_nodes):
            for core_id in range(self.num_cores_per_node):
                units.append(f"{node_id}:C{core_id}")
                units.append(f"{node_id}:B{core_id}")

        comm_matrix = {row: {col: 0 for col in units} for row in units}
        for comm in traffic:
            src = comm["src"]
            dsts = comm["dst"].split(",")
            for dst in dsts:
                comm_matrix[src][dst] += comm["size"]
        
        comm_df = pd.DataFrame(comm_matrix)
        comm_df.to_csv(fname)

        return comm_matrix