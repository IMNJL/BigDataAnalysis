def log_message(message):
    print(f"[LOG] {message}")

def visualize_data(data, title="Data Visualization"):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(title)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid()
    plt.show()