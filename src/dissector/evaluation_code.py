import SimpleITK as sitk

# Read 3D images (e.g., .nii, .mha, .nrrd)
fixed = sitk.ReadImage("fixed.nii", sitk.sitkFloat32)
moving = sitk.ReadImage("moving.nii", sitk.sitkFloat32)

# Initialize transform (important!)
initial_transform = sitk.CenteredTransformInitializer(
    fixed,
    moving,
    sitk.Euler3DTransform(),
    sitk.CenteredTransformInitializerFilter.GEOMETRY
)

# Registration setup
registration_method = sitk.ImageRegistrationMethod()

# Similarity metric (great for multi-modal)
registration_method.SetMetricAsMattesMutualInformation(50)

# Sampling (speeds things up for large volumes)
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.01)

# Interpolator
registration_method.SetInterpolator(sitk.sitkLinear)

# Optimizer
registration_method.SetOptimizerAsGradientDescent(
    learningRate=1.0,
    numberOfIterations=200,
    convergenceMinimumValue=1e-6,
    convergenceWindowSize=10
)

registration_method.SetOptimizerScalesFromPhysicalShift()

# Multi-resolution (VERY important for 3D)
registration_method.SetShrinkFactorsPerLevel([4, 2, 1])
registration_method.SetSmoothingSigmasPerLevel([2, 1, 0])
registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

# Set initial transform
registration_method.SetInitialTransform(initial_transform, inPlace=False)

# Run registration
final_transform = registration_method.Execute(fixed, moving)

print("Final metric value:", registration_method.GetMetricValue())
print("Optimizer stop condition:", registration_method.GetOptimizerStopConditionDescription())

# Apply transform
resampled = sitk.Resample(
    moving,
    fixed,
    final_transform,
    sitk.sitkLinear,
    0.0,
    moving.GetPixelID()
)

# Save result
sitk.WriteImage(resampled, "aligned.nii")