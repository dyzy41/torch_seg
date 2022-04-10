import nibabel as nib

example_filename = r'E:\0Ecode\code220321_hoodyseg\TrainingData\12m\012m_044a.v2_outskull_mask.nii.gz'
img = nib.load(example_filename)
img_affine = img.affine
img = img.get_data()

print('ok')
