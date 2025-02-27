import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_sd19_model(input_shape=(128, 128, 1), num_classes=47):
    model = models.Sequential([
        # Feature extraction
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Classification
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model






# SD19 directory structure (after unzipping)
data_dir = 'by_merge'

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



model = create_sd19_model()

history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5),
        tf.keras.callbacks.ModelCheckpoint('sd19_model.h5', save_best_only=True)
    ]
)


# Merged class mapping (47 classes)
class_mapping = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C.c', 'D', 'E', 'F', 'G', 'H', 'I.i', 'J.j',
    'K.k', 'L.l', 'M.m', 'N', 'O.o', 'P.p', 'Q', 'R', 'S.s',
    'T', 'U.u', 'V.v', 'W.w', 'X.x', 'Y.y', 'Z.z',
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]