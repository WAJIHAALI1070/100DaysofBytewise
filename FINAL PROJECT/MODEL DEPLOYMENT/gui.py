import tkinter as tk
from tkinter import messagebox


def submit_transaction():
    # Retrieve the data from the input fields
    training_config = training_config_entry.get()
    metrics = metrics_entry.get()
    comments = comments_entry.get("1.0", tk.END)

    # Process or store the data as needed
    print(f"Training Configuration: {training_config}")
    print(f"Metrics: {metrics}")
    print(f"Comments: {comments}")

    # Show a confirmation message
    messagebox.showinfo("Submission", "Transaction details submitted successfully!")


# Create the main window
root = tk.Tk()
root.title("Transaction Details")

# Create and place input fields
tk.Label(root, text="Training Configuration:").grid(row=0, column=0, padx=10, pady=5)
training_config_entry = tk.Entry(root, width=50)
training_config_entry.grid(row=0, column=1, padx=10, pady=5)

tk.Label(root, text="Metrics:").grid(row=1, column=0, padx=10, pady=5)
metrics_entry = tk.Entry(root, width=50)
metrics_entry.grid(row=1, column=1, padx=10, pady=5)

tk.Label(root, text="Additional Comments:").grid(row=2, column=0, padx=10, pady=5)
comments_entry = tk.Text(root, width=50, height=10)
comments_entry.grid(row=2, column=1, padx=10, pady=5)

# Create and place the submit button
submit_button = tk.Button(root, text="Submit", command=submit_transaction)
submit_button.grid(row=3, column=1, padx=10, pady=10, sticky=tk.E)

# Run the application
root.mainloop()
