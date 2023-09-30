from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from keras.layers import Input, Dense, Flatten
from keras.models import Model

# Veri yolu
train_dir = 'data/'

# Veri zenginleştirme
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Veri yükleyici
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

# Pretrained model yükleme
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Fully connected layer
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# Model oluşturma
model = Model(inputs=base_model.input, outputs=predictions)

# Convolutional katmanları dondurma
for layer in base_model.layers:
    layer.trainable = False

# Modeli derleme
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model eğitimi
model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10)
    
# Modeli kaydetme
model.save('model.h5')
print('Done!')