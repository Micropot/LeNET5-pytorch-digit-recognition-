from tkinter import  filedialog

def upload():
    # ask the user for the file he wants to upload
    file_path = filedialog.askopenfilename()

    return str(file_path)
