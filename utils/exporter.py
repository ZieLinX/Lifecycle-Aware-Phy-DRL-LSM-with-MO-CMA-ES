import omni.kit.asset_converter as converter
import trimesh
import os

async def export_results(usd_path, output_name):
    # 1. 导出 STL (网格数据)
    conv = converter.get_instance()
    settings = converter.AssetConverterSettings()
    stl_path = os.path.abspath(f"{output_name}.stl")
    
    task = conv.create_converter_task(usd_path, stl_path, None, settings)
    await task.wait_until_finished()
    print(f"STL exported: {stl_path}")

    # 2. 导出 STEP (通过 trimesh 转存再处理)
    # 注意：需要安装 trimesh 和 pythonocc-core 或使用 FreeCAD 命令行
    mesh = trimesh.load(stl_path)
    if mesh.is_watertight:
        # 这里仅为逻辑占位，实际 STEP 需 B-Rep 封装
        print("STEP Export: Recommended using FreeCAD-Python API to convert STL to STEP.")