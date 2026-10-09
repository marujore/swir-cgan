"""Usage examples for wfi-swir-cgan."""

from wfi_swir_cgan import SWIRGenerator, generate_swir1, generate_swir2

# 1) Single 4-band stack (band order: blue, green, red, nir)
generate_swir1(input_raster='input/wfi_stack.tif', output_raster='output/wfi_B11.tif', batch_size=16)
generate_swir2(input_raster='input/wfi_stack.tif', output_raster='output/wfi_B12.tif')  # batch_size defaults to 32

# 2) Four separate single-band files, as a list in the order [blue, green, red, nir]
bands = ['input/wfi_blue.tif', 'input/wfi_green.tif', 'input/wfi_red.tif', 'input/wfi_nir.tif']
generate_swir1(bands, 'output/wfi_B11.tif')

# 3) Four separate files as a dictionary (order does not matter)
bands = {
    'blue': 'input/wfi_blue.tif',
    'green': 'input/wfi_green.tif',
    'red': 'input/wfi_red.tif',
    'nir': 'input/wfi_nir.tif',
}
generate_swir2(bands, 'output/wfi_B12.tif')

# 4) Processing many scenes: load the model once and reuse it
generator = SWIRGenerator('B11', device='cuda', batch_size=32)
for scene in ['scene_a.tif', 'scene_b.tif']:
    generator.predict(f'input/{scene}', f'output/B11_{scene}')
