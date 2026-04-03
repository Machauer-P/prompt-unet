import tensorflow as tf
import tensorflowjs as tfjs
import argparse
import os

def export_model(input_model_path, output_tfjs_dir):
    if not os.path.exists(input_model_path):
        print(f"Error: Could not find model at '{input_model_path}'")
        return

    print(f"Loading model from {input_model_path}...")
    # If there are custom layers, you might need to pass custom_objects={'YourLayer': YourLayer} here
    model = tf.keras.models.load_model(input_model_path)

    print(f"Exporting to TensorFlow.js format in '{output_tfjs_dir}/'...")
    tfjs.converters.save_keras_model(model, output_tfjs_dir)
    
    print("Export complete! Your files are ready.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a Keras model to TensorFlow.js format.")
    
    # Defaulting to the same filenames as in the original notebook 
    # but allowing them to be overridden from the command line for better reusability!
    parser.add_argument("--input", "-i", type=str, default="p_unet21.keras", help="Path to input .keras model file")
    parser.add_argument("--output", "-o", type=str, default="p_unet21_tfjs", help="Directory to save output tfjs model")
    
    args = parser.parse_args()
    
    export_model(args.input, args.output)
