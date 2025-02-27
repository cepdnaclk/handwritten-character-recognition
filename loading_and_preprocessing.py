from keras.preprocessing.image import ImageDataGenerator

# SD19 directory structure (after unzipping)
data_dir = 'D:\semester 5\CO542-2024  Neural Networks and Fuzzy Systems\project\data\by_merge'

# Create data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    color_mode='grayscale',
    batch_size=128,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    color_mode='grayscale',
    batch_size=128,
    class_mode='categorical',
    subset='validation'
)
