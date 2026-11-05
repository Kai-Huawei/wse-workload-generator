"""
mapping_log.py - record, for every layer that produces core-level traces,
  1) where its tensors are placed in memory   (one row per tile)
  2) which core each of its operations runs on (one row per operation)

Output is split by node and by iteration (like the traces: traces/decode/node_0/decode0/):

    output/mapping/node_0/decode0/memory.csv
    output/mapping/node_0/decode0/cores.csv
    output/mapping/node_1/decode0/memory.csv
    ...

Called from generate_traces.py right after each layer is built:

    from src.core_level.common.mapping_log import set_output_files, log_layer

    set_output_files(memory_csv=..., core_csv=..., iteration=args.iter)   # once, at the start
    ...
    layer = layer_class(**layer_params)
    log_layer(layer, uid, node_id)                   # after each layer

The paths given to set_output_files act as templates: "output/mapping/memory.csv"
becomes "output/mapping/node_<id>/<iteration>/memory.csv" (no iteration set: no
iteration folder). Set PER_NODE = False for one file per iteration for all nodes
("output/mapping/<iteration>/memory.csv").

The layer files themselves (linear.py, mla.py, ...) are not modified: the
logger reads the finished layer object. Each CSV is overwritten the first time
it is written in a run, then appended to.
"""
import csv
import os

from src.node_level.common.utils import dtype_to_byte
from src.core_level.common.tensor import Tensor
from src.core_level.common.tile import Tile

MEMORY_MAP_CSV = "output/mapping/memory.csv"
CORE_MAP_CSV = "output/mapping/cores.csv"
PER_NODE = True   # True: one folder per node. False: one big file for all nodes.
ITERATION = None  # e.g. "decode0" (generate_traces.py --iter): a folder per iteration; None = no folder

# Every CSV path already written in this run. A path not in here is opened
# with "w" (overwriting old results) and gets a header; afterwards "a".
_started_paths = set()

# For each kind of tile operation: its short type name, and which of its
# attributes hold the tiles it reads (input), reads as weight-like data, and
# writes (output). For attention, the KV tile goes in the weight column.
OP_TILES = {
    "TileGemmOp":            ("gemm",             ["input_tile"], ["weight_tile"], ["out_tile"]),
    "TileGroupGemmOp":       ("gemm",             ["input_tile"], ["weight_tile"], ["out_tile"]),
    "TileAttnOp":            ("attention",        ["q_tile"],     ["kv_tile"],     ["out_tile"]),
    "TileReduceOp":          ("reduce",           ["in_tiles"],   [],              ["out_tile"]),
    "TileUnicastOp":         ("unicast",          ["input_tile"], [],              ["out_tile"]),
    "TileMulticastOp":       ("multicast",        ["input_tile"], [],              ["out_tiles"]),
    "TileAllreduceStage1Op": ("allreduce_stage1", ["send0_tile"], [],              ["next_tile"]),
    "TileAllreduceStage2Op": ("allreduce_stage2", ["send_tile"],  [],              ["next_tile"]),
}

MEMORY_HEADER = ["uid", "node_id", "tensor", "role", "tile_index", "element_range", "bytes", "bank"]
CORE_HEADER = ["uid", "node_id", "op_type", "core_id", "input_bank", "weight_bank", "output_bank",
               "input_bytes", "weight_bytes", "output_bytes"]


def set_output_files(memory_csv=None, core_csv=None, per_node=None, iteration=None):
    """Optionally change where the CSV files are written, whether they are split by node, and the
    iteration (e.g. "decode0"): the files go into a folder named after it."""
    global MEMORY_MAP_CSV, CORE_MAP_CSV, PER_NODE, ITERATION
    if memory_csv is not None:
        MEMORY_MAP_CSV = memory_csv
    if core_csv is not None:
        CORE_MAP_CSV = core_csv
    if per_node is not None:
        PER_NODE = per_node
    if iteration is not None:
        ITERATION = iteration
    _started_paths.clear()


def _node_path(template, node_id):
    """'output/mapping/memory.csv', 3 -> 'output/mapping/node_3/decode0/memory.csv' (ITERATION "decode0")."""
    folder, name = os.path.split(template)
    if PER_NODE:
        folder = os.path.join(folder, f"node_{node_id}")
    if ITERATION:
        folder = os.path.join(folder, ITERATION)
    return os.path.join(folder, name)


def _open_csv(path, header):
    """Open the CSV: 'w' + header the first time in this run, 'a' afterwards."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    started = path in _started_paths
    f = open(path, "a" if started else "w", newline="")
    writer = csv.writer(f)
    if not started:
        writer.writerow(header)
        _started_paths.add(path)
    return f, writer


def _leaves(value):
    """Yield every item inside nested dicts / lists / tuples (or the value itself)."""
    if isinstance(value, dict):
        for v in value.values():
            yield from _leaves(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _leaves(v)
    else:
        yield value


def _role(attr_name):
    """'weight_tensor' -> 'weight', 'partial_sum_tiles' -> 'partial_sum', ..."""
    for suffix in ("_tensors", "_tensor", "_tiles", "_tile"):
        if attr_name.endswith(suffix):
            return attr_name[: -len(suffix)]
    return attr_name


# ----------------------------------------------------------------------------
# 1) Memory placement
# ----------------------------------------------------------------------------
def _tile_leaves(memory_map, index=()):
    """Yield (tile_index, tile_entry) for every tile of a tensor's memory map.

    tile_entry is {"range": [(start, end), ...], "bank": MemoryBank}.
    """
    if "bank" in memory_map:
        yield index, memory_map
        return
    for key in memory_map:
        yield from _tile_leaves(memory_map[key], index + (key,))


def _layer_tensors(layer):
    """(tensor, role) for every placed tensor the layer holds.

    Tensors stored directly on the layer come first; tensors reachable only
    through tiles (e.g. partial sums) are found through each tile's parent.
    """
    seen, found = set(), []
    attrs = list(vars(layer).items())
    for want_tiles in (False, True):
        for name, value in attrs:
            for item in _leaves(value):
                if want_tiles and isinstance(item, Tile):
                    tensor = item.parent
                elif not want_tiles and isinstance(item, Tensor):
                    tensor = item
                else:
                    continue
                if id(tensor) in seen or tensor.memory_map is None:
                    continue
                seen.add(id(tensor))
                found.append((tensor, _role(name)))
    return found


def _write_memory_rows(writer, uid, node_id, tensor, role):
    for tile_index, tile in _tile_leaves(tensor.memory_map):
        n_elements = 1
        for start, end in tile["range"]:
            n_elements *= (end - start)
        writer.writerow([uid, node_id, tensor.uid, role,
                         ",".join(str(i) for i in tile_index),                          # e.g. 1,0
                         " ".join(f"{start}:{end}" for start, end in tile["range"]),    # e.g. 448:896 0:288
                         n_elements * dtype_to_byte(tensor.prec),                       # bytes in this tile
                         tile["bank"].bank_id])


def log_memory_mapping(uid, node_id, tensor, role):
    """Log one tensor (one row per tile). Can also be called on its own."""
    f, writer = _open_csv(_node_path(MEMORY_MAP_CSV, node_id), MEMORY_HEADER)
    with f:
        _write_memory_rows(writer, uid, node_id, tensor, role)


# ----------------------------------------------------------------------------
# 2) Core assignment
# ----------------------------------------------------------------------------
def _layer_ops(layer):
    """Every operation the layer holds that has been assigned to a core."""
    seen, ops = set(), []
    for _, value in vars(layer).items():
        for item in _leaves(value):
            if getattr(item, "mapped_core", None) is not None and id(item) not in seen:
                seen.add(id(item))
                ops.append(item)
    return ops


def _banks(op, attr_names):
    """Global bank ids holding the tiles in the given attributes, joined with '|'."""
    banks = []
    for name in attr_names:
        for tile in _leaves(getattr(op, name, None)):
            if tile is None:
                continue
            for bank in tile.get_physical_address():
                if bank.bank_id not in banks:
                    banks.append(bank.bank_id)
    return "|".join(str(b) for b in banks)


def _bytes(op, attr_names):
    """Total bytes held by the tiles in the given attributes ('' if there are none)."""
    total, found = 0, False
    for name in attr_names:
        for tile in _leaves(getattr(op, name, None)):
            if tile is None:
                continue
            found = True
            total += sum(tile.get_physical_address().values())
    return total if found else ""


def _write_core_row(writer, uid, node_id, op):
    op_type, ins, weights, outs = OP_TILES.get(type(op).__name__, (type(op).__name__, [], [], []))
    writer.writerow([uid, node_id, op_type, op.mapped_core.core_id,
                     _banks(op, ins), _banks(op, weights), _banks(op, outs),
                     _bytes(op, ins), _bytes(op, weights), _bytes(op, outs)])


def log_core_mapping(uid, node_id, op):
    """Log one operation. Can also be called on its own."""
    f, writer = _open_csv(_node_path(CORE_MAP_CSV, node_id), CORE_HEADER)
    with f:
        _write_core_row(writer, uid, node_id, op)


# ----------------------------------------------------------------------------
# One call per layer
# ----------------------------------------------------------------------------
def log_layer(layer, uid, node_id):
    """Log all tensors and all operations of one finished layer into that node's folder."""
    f, writer = _open_csv(_node_path(MEMORY_MAP_CSV, node_id), MEMORY_HEADER)
    with f:
        for tensor, role in _layer_tensors(layer):
            _write_memory_rows(writer, uid, node_id, tensor, role)

    f, writer = _open_csv(_node_path(CORE_MAP_CSV, node_id), CORE_HEADER)
    with f:
        for op in _layer_ops(layer):
            _write_core_row(writer, uid, node_id, op)