import tkinter as tk

root= tk.Tk()

canvas1 = tk.Canvas(root, width = 400, height = 300)
canvas1.pack()

entry1 = tk.Entry (root) 
canvas1.create_window(200, 140, window=entry1)

def getSquareRoot (): 
    exec(open("macrious_adera_test_19a.py").read()) 
    label1 = tk.Label(root, text= "the answer is NOT always paracetamol..file has been successfully written")
    canvas1.create_window(200, 230, window=label1)
    

    
button1 = tk.Button(text='Enter your question', command=getSquareRoot)
canvas1.create_window(200, 180, window=button1)

root.mainloop()