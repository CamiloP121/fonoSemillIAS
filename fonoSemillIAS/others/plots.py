import matplotlib.pyplot as plt

def simple_plot(arr, title, ylabel, xlabel):
  fig, ax = plt.subplots(figsize=(20,3))
  ax.plot(arr)
  ax.set_ylabel(ylabel); ax.set_title(title) ; ax.set_xlabel(xlabel)
  plt.show()