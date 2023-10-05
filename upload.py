from tkinter import messagebox, simpledialog, filedialog

def upload():
    file_path = filedialog.askopenfilename()

    return str(file_path)
