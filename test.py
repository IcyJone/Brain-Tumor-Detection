import requests
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define the Flask API URL
url = "http://127.0.0.1:5000/predict"  # Change if your API is running on a different host or port

# Function to ask user for the image path
def get_image_path():
    # Ask for the image path
    image_path = input("Please enter the full path of the image: ")
    return image_path

# Function to display image with prediction result
def display_image_with_prediction(image_path, result):
    # Load and display the image
    img = mpimg.imread(image_path)
    plt.figure(figsize=(10, 8))  # Larger figure for better presentation
    plt.imshow(img)
    plt.axis('off')  # Remove axis for a cleaner look

    # Correctly access the 'Accuracy' and 'prediction' keys
    accuracy = result.get('Accuracy', "N/A")  # Fallback if accuracy is not provided

    # Set a large, bold title with a font size of 20 and a clear prediction message
    plt.title(f"Prediction: {result['prediction']}\nAccuracy: {accuracy*100:.2f}%" if isinstance(accuracy, float) else f"Prediction: {result['prediction']}\nAccuracy: {accuracy}", fontsize=20, fontweight='bold')

    # Display the image with the prediction title
    plt.show()

# Main logic
def main():
    # Get the image path from the user
    image_path = get_image_path()

    try:
        # Open and send the image to the API for prediction
        with open(image_path, "rb") as image_file:
            response = requests.post(url, files={"file": image_file})

        # Check the response status
        if response.status_code == 200:
            result = response.json()  # Assuming the API returns a JSON response with the prediction
            print(result)  # For debugging, print the full response to understand the structure
            display_image_with_prediction(image_path, result)
        else:
            print(f"Error: Received status code {response.status_code}")
            print(response.text)

    except FileNotFoundError:
        print(f"Error: File not found at {image_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the script
if __name__ == "__main__":
    main()
