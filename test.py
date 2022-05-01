# a = []
import pydicom

filename = 'MOLECUBE/20220427141045_CT_ISRA_0_mouse_alive.dcm'
ds = pydicom.dcmread(filename)
new_image = ds.pixel_array.astype(float)
print(new_image.shape)