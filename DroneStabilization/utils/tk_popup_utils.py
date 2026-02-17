import tkinter as tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


'''create pop up window for viewing and selecting segmented image'''
def pop_up(fig, subroot):
    canvas = FigureCanvasTkAgg(fig, master=subroot)
    canvas.draw()

    toolbar = NavigationToolbar2Tk(canvas, subroot, pack_toolbar=False)
    toolbar.update()

    canvas.mpl_connect(
        "key_press_event", lambda event: print(f"you pressed {event.key}"))
    canvas.mpl_connect("key_press_event", key_press_handler)

    button_quit = tk.Button(master=subroot, text="Quit", command=subroot.destroy)
    return [button_quit, toolbar, canvas]


'''toolbar functions for selecting image'''


def image_selection(subroot, n_cluster, command, canvas, start=0, end=0):
    options = list(range(start, n_cluster+end))
    # convert to strings
    options = [str(x) for x in options]
    #
    variable = tk.StringVar(subroot)
    variable.set(options[0])
    selector = tk.OptionMenu(subroot, variable, *options, command=command)
    canvas[0].pack(side=tk.BOTTOM)
    selector.pack(side=tk.BOTTOM)
    canvas[1].pack(side=tk.BOTTOM, fill=tk.X)
    canvas[2].get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
