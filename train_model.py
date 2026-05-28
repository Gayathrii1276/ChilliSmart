import argparse
import json
import os
import tensorflow as tf
from tensorflow.keras import layers, models


def build_simple_cnn(input_shape, num_classes):
	inputs = layers.Input(shape=input_shape)
	x = layers.Rescaling(1.0 / 255)(inputs)
	x = layers.Conv2D(32, 3, activation="relu")(x)
	x = layers.MaxPooling2D()(x)
	x = layers.Conv2D(64, 3, activation="relu")(x)
	x = layers.MaxPooling2D()(x)
	x = layers.Conv2D(128, 3, activation="relu")(x)
	x = layers.MaxPooling2D()(x)
	x = layers.Flatten()(x)
	x = layers.Dense(128, activation="relu")(x)
	x = layers.Dropout(0.4)(x)
	outputs = layers.Dense(num_classes, activation="softmax")(x)
	model = models.Model(inputs, outputs)
	return model


def main(args):
	data_dir = args.data_dir
	img_size = (args.img_size, args.img_size)
	batch_size = args.batch_size

	if not os.path.isdir(data_dir):
		raise SystemExit(f"Dataset directory not found: {data_dir}")

	train_ds = tf.keras.preprocessing.image_dataset_from_directory(
		data_dir,
		labels="inferred",
		label_mode="categorical",
		batch_size=batch_size,
		image_size=img_size,
		shuffle=True,
		validation_split=0.2,
		subset="training",
		seed=123,
	)

	val_ds = tf.keras.preprocessing.image_dataset_from_directory(
		data_dir,
		labels="inferred",
		label_mode="categorical",
		batch_size=batch_size,
		image_size=img_size,
		shuffle=True,
		validation_split=0.2,
		subset="validation",
		seed=123,
	)

	class_names = train_ds.class_names
	num_classes = len(class_names)
	print(f"Found classes: {class_names}")

	AUTOTUNE = tf.data.AUTOTUNE
	train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
	val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

	model = build_simple_cnn(input_shape=img_size + (3,), num_classes=num_classes)
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="categorical_crossentropy", metrics=["accuracy"]) 

	callbacks = [
		tf.keras.callbacks.ModelCheckpoint(args.output_model, save_best_only=True, monitor="val_loss"),
		tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
	]

	history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

	# final save
	model.save(args.output_model)

	# write class names
	with open(args.labels_out, "w") as f:
		json.dump(class_names, f)

	print(f"Model saved to {args.output_model}")
	print(f"Labels saved to {args.labels_out}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train a small CNN on the chili dataset")
	parser.add_argument("--data-dir", default="dataset", help="Path to dataset root (contains class subfolders)")
	parser.add_argument("--img-size", type=int, default=224, help="Image size (square)")
	parser.add_argument("--batch-size", type=int, default=16)
	parser.add_argument("--epochs", type=int, default=30)
	parser.add_argument("--output-model", default="mirchi_model.h5", help="Path to save the trained model")
	parser.add_argument("--labels-out", default="labels.json", help="Path to save class names mapping")
	args = parser.parse_args()
	main(args)


