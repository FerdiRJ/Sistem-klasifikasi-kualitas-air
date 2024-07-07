from tkinter import Tk, Frame, Button, PhotoImage, Label, IntVar, ttk, messagebox, Toplevel
from tkinter import Tk, Frame, Entry, Button, PhotoImage, Label
from tkinter import ttk
import joblib
from simulated_annealing_knn import SimulatedAnnealingKNN
from tkinter import messagebox

class OpeningPage:
    def __init__(self, master):
        self.master = master
        self.master.title("Opening Page")

        self.master.geometry("1280x823")

        # Load background image for opening page
        bg_image_opening_page = PhotoImage(file="bgbgbg1.png")

        # Create a frame for the opening page background
        self.bg_frame_opening_page = Frame(self.master, width=1280, height=832)
        self.bg_frame_opening_page.place(x=0, y=0)

        # Display the opening page background image
        self.bg_label_opening_page = Label(self.bg_frame_opening_page, image=bg_image_opening_page)
        self.bg_label_opening_page.image = bg_image_opening_page
        self.bg_label_opening_page.place(x=0, y=0, relwidth=1, relheight=1)  # Cover the entire window


        # open_button = Button(self.master, text="Deteksi Sekarang", command=self.open_main_gui)
        # open_button.place(x=1111, y=60)

        # Load button image
        self.button_image_1 = PhotoImage(file="deteksinow.png")

        # Create and display the button
        self.button_1 = Button(image=self.button_image_1, borderwidth=0, highlightthickness=0, command=self.open_main_gui, relief="flat")
        self.button_1.image = self.button_image_1  # Keep a reference to avoid garbage collection
        self.button_1.place(x=1050, y=33)


    def open_main_gui(self):
        self.master.destroy()  # Close the opening page
        root = Tk()
        my_gui = MyGUI(root)
        root.mainloop()


class MyGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("My GUI")
        self.master.title("Full Screen Image")
        self.master.geometry("1280x832")

        # Load background image
        bg_image = PhotoImage(file="MacBook Air.png")

        # Create a frame for the background
        self.bg_frame = Frame(self.master, width=1280, height=832)
        self.bg_frame.place(x=0, y=0)

        # Display the background image
        self.bg_label = Button(self.bg_frame, image=bg_image, borderwidth=0, command=self.submit_clicked)
        self.bg_label.image = bg_image
        self.bg_label.place(x=0, y=0)

        # Create Entry widgets
        self.sliders = []
        self.slider_labels = []

        slider_positions = [
            ("PH", 56, 285, 1, 14),
            ("Padatan", 866, 285, 23, 61227),
            ("Kesadahan", 461, 285, 47, 323),
            ("Kloramina", 56, 445, 1, 13),
            ("Sulfat", 461, 445, 129, 481),
            ("Konduktivitas", 866, 445, 181, 753),
            ("Karbon Organik", 56, 605, 2, 28),
            ("Trihalomethane", 461, 605, 0, 124),
            ("Kekeruhan", 866, 605, 1, 6)
        ]

        for i, (name, x, y, min_val, max_val) in enumerate(slider_positions):
            label = Label(self.bg_frame, text=name, font=('Poppins', 25, 'bold'), fg='#000080', bg="white")
            label.place(x=x, y=y - 30)

            slider_var = IntVar()  # Menggunakan IntVar untuk bilangan bulat
            slider = ttk.Scale(self.bg_frame, from_=min_val, to=max_val, orient="horizontal", variable=slider_var,
                               length=300,
                               command=lambda val, idx=i, slider_var=slider_var, label=label: self.update_result_label(
                                   val, idx, slider_var, label))
            slider.place(x=x, y=y)
            self.sliders.append(slider)

        # Create Label to display prediction result
        self.prediction_label = Label(self.bg_frame, text="", font=('Poppins', 20, 'bold'), fg='#000080', bg="white")
        self.prediction_label.place(x=300, y=700)

        # # Create Button widgets
        # self.submit_button = Button(self.bg_frame, text="Periksa", command=self.submit_clicked, font=('Poppins', 14, 'bold'),
        #                             fg='#000080', bg= "#448AC7")
        # self.submit_button.place(x=400, y=700, width=100, height=40)

        self.button_image_1 = PhotoImage(file="Group 15.png")

        # Create and display the button
        self.button_1 = Button(image=self.button_image_1, borderwidth=0, highlightthickness=0, command=self.submit_clicked, relief="flat")
        self.button_1.image = self.button_image_1  # Keep a reference to avoid garbage collection
        self.button_1.place(x=150, y=700)

        # self.additional_page_button = Button(self.bg_frame, text="Additional Page", command=self.submit_clicked,
        #                                      font=('Arial', 14, 'bold'), fg='#000080')
        # self.additional_page_button.place(x=350, y=700, width=150, height=40)
        
        # Load your model
        self.model = joblib.load("knn_optimasimodel.pkl")

    def submit_clicked(self):
        # Perform actions when the Submit button is clicked
        slider_values = [slider.get() for slider in self.sliders]
        self.update_prediction(None, None, None, None)  # Update prediction result
        result = self.predict(*slider_values)
        self.display_prediction(result)

    def button_clicked(self):
        # Define actions when the background image is clicked
        print("Background Clicked")

    def update_result_label(self, value, index, slider_var, label):
        label.config(text=f"{label['text'].split(':')[0]}: {slider_var.get()}")

    def predict(self, PH, Padatan, Kesadahan, Kloramina, Sulfat, Konduktivitas, Karbon_Organik, Trihalomethane, Kekeruhan):
        sc = self.model["scaler"]
        clf = self.model["model"]

        # Use the scaler on the input data
        xtest = sc.transform([[PH, Padatan, Kesadahan, Kloramina, Sulfat, Konduktivitas, Karbon_Organik, Trihalomethane, Kekeruhan]])

        # Make prediction using the model
        prediction = clf.predict(xtest)
        return prediction

    def update_prediction(self, value, index, slider_var, label):
        # Update the prediction result in real-time
        slider_values = [slider.get() for slider in self.sliders]
        result = self.predict(*slider_values)
        self.display_prediction(result)

    def display_prediction(self, result):
        # Map the result
        if result == 1:
            mapped_result = "Water is safe for human consumption"
        elif result == 0:
            mapped_result = "Water is not safe for human consumption"
        else:
            mapped_result = "Invalid prediction result"

        # Display the result using Label
        self.prediction_label.config(text=mapped_result)



if __name__ == "__main__":
    root = Tk()
    opening_page = OpeningPage(root)
    root.mainloop()   