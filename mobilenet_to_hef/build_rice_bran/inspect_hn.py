from hailo_sdk_client import ClientRunner

runner = ClientRunner(har="build_rice_bran/rice_bran_parsed.har")
layers = runner.get_hn_dict()["layers"]
for name, layer in layers.items():
    if "pool" in layer["type"] or "avg" in name:
        print(name, layer["type"], layer.get("input_shapes"), layer.get("output_shapes"))
print("types:", sorted({layer["type"] for layer in layers.values()}))
