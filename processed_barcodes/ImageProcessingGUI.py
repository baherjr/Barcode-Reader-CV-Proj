import os
import customtkinter as ctk  # Import customtkinter for modern design
from tkinter import filedialog
from PIL import Image, ImageTk
from ImageProcessor import ImageProcessor  # Assuming the class is implemented

class ImageProcessingGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configure Main Window
        self.title("Image Processing Application")
        self.geometry("1000x700")
        self.configure(fg_color='#00468c')  # Light blue background

        # Configure Initial Appearance (Dark/Light)
        ctk.set_appearance_mode("system")  # Options: "light", "dark", "system" (follows system theme)
        ctk.set_default_color_theme("dark-blue")  # Options: "blue", "green", "dark-blue"

        # Set grid weights to make the window resizable
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)  # Makes the middle section (canvases) flexible

        # Variables to hold image paths
        self.image_path = None
        self.processed_image_path = None

        # Create Widgets
        self.create_widgets()

    def create_widgets(self):
        # Upload Button at the top
        self.upload_button = ctk.CTkButton(
            master=self,
            text="Upload Image",
            command=self.upload_image,
            corner_radius=20,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#002851"
        )
        self.upload_button.grid(row=0, column=0, pady=20, padx=10, sticky="ew")  # Expand button horizontally

        # Main Frame for Canvases (Middle Section)
        self.frame_images = ctk.CTkFrame(self)
        self.frame_images.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")

        # Configure the grid inside the frame to make the canvases resizable
        self.frame_images.grid_columnconfigure(0, weight=1)  # Original canvas resizable
        self.frame_images.grid_columnconfigure(1, weight=1)  # Processed canvas resizable
        self.frame_images.grid_rowconfigure(1, weight=1)  # Vertically resizable

        # Canvases for displaying images
        self.canvas_original_label = ctk.CTkLabel(self.frame_images, text="Original Image", font=ctk.CTkFont(size=12))
        self.canvas_original_label.grid(row=0, column=0, pady=10)

        self.canvas_original = ctk.CTkCanvas(self.frame_images, bg="white", highlightthickness=2)
        self.canvas_original.grid(row=1, column=0, padx=15, pady=10, sticky="nsew")

        self.canvas_processed_label = ctk.CTkLabel(self.frame_images, text="Processed Image", font=ctk.CTkFont(size=12))
        self.canvas_processed_label.grid(row=0, column=1, pady=10)

        self.canvas_processed = ctk.CTkCanvas(self.frame_images, bg="white", highlightthickness=2)
        self.canvas_processed.grid(row=1, column=1, padx=15, pady=10, sticky="nsew")

        # Results Section (Bottom)
        self.results_frame = ctk.CTkFrame(self, corner_radius=10)
        self.results_frame.grid(row=2, column=0, pady=20, padx=10, sticky="ew")

        # Configure results frame grid to make it flexible
        self.results_frame.grid_columnconfigure(0, weight=1)

        self.flags_label = ctk.CTkLabel(self.results_frame, text="Flags Triggered: None", anchor="w",
                                        font=ctk.CTkFont(size=12, weight="bold"))
        self.flags_label.grid(row=0, column=0, sticky="w", padx=10, pady=5)

        self.barcode_label = ctk.CTkLabel(self.results_frame, text="Decoded Barcode: None", anchor="w",
                                          font=ctk.CTkFont(size=12, weight="bold"))
        self.barcode_label.grid(row=1, column=0, sticky="w", padx=10, pady=5)

    def upload_image(self):
        # Open file dialog to select an image
        self.image_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )

        if not self.image_path:
            return

        # Display the selected image on the left canvas
        self.display_image(self.image_path, self.canvas_original)

        # Process the image
        self.process_image()

    def process_image(self):
        # Ensure an image is selected
        if not self.image_path:
            return

        # Process the image using ImageProcessor
        processor = ImageProcessor([self.image_path])
        processor.process_image(self.image_path)

        # Retrieve the processed image path
        self.processed_image_path = os.path.join(ImageProcessor.OUTPUT_PATH, os.path.basename(self.image_path))

        # Display the processed image on the right canvas
        if os.path.exists(self.processed_image_path):
            self.display_image(self.processed_image_path, self.canvas_processed)

        # Display the results (flags and barcode)
        self.display_results(self.image_path)

    def display_image(self, image_path, canvas):
        # Load and display image on the specified canvas
        image = Image.open(image_path)
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        image.thumbnail((canvas_width, canvas_height))  # Resize image dynamically to fit the canvas
        photo = ImageTk.PhotoImage(image)

        # Clear canvas content and display new image
        canvas.delete("all")
        canvas.image = photo  # Keep a reference to avoid garbage collection
        canvas.create_image(canvas_width / 2, canvas_height / 2, image=photo)

    def display_results(self, image_path):
        # Path to the file where results (flags and barcode) are saved
        flags_and_decoded_path = os.path.join(ImageProcessor.OUTPUT_PATH, "flags_and_decoded.txt")
        image_name = os.path.basename(image_path)

        # Default results
        flags_text = "Flags Triggered: No flags detected"
        decoded_text = "Decoded Barcode: None"

        # Read and extract results from the file, if available
        if os.path.exists(flags_and_decoded_path):
            with open(flags_and_decoded_path, "r") as file:
                for line in file:
                    if line.startswith(image_name):
                        parts = line.strip().split(": Flags: ")
                        if len(parts) > 1:
                            flags_and_decoded = parts[1].split(", Decoded: ")
                            flags_text = f"Flags Triggered: {flags_and_decoded[0]}"
                            decoded_text = f"Decoded Barcode: {flags_and_decoded[1]}" if len(
                                flags_and_decoded) > 1 else "Decoded Barcode: None"
                        break

        # Update the results in the GUI
        self.flags_label.configure(text=flags_text)
        self.barcode_label.configure(text=decoded_text)

# Main Application
if __name__ == "__main__":
    # Ensure the output directory exists
    if not os.path.exists(ImageProcessor.OUTPUT_PATH):
        os.makedirs(ImageProcessor.OUTPUT_PATH)

    app = ImageProcessingGUI()
    app.mainloop()