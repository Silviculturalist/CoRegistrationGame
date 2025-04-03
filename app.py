import os
import sys
import queue
import threading
import tkinter as tk
import pygame
import logging
from pynput import keyboard
import pandas as pd
import numpy as np
from copy import deepcopy

# Import custom modules
from trees import Stand, SavedStand
from chm_plot import CHMPlot, SavedPlot, PlotCenters, to_screen_coordinates, get_viewport_scale, draw_plot, draw_chm, draw_polygon, is_point_in_polygon
from ficp import FractionalICP

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
TRANSLATE_STEP = 0.5
PAN_STEP = 5
ZOOM_STEP = 0.3
TREE_SCALE_INITIAL = 1.0

class App:
    def __init__(self, root, plot_stand, chm_stand, plot_centers):
        self.root = root
        self.plot_stand = plot_stand
        self.chm_stand = chm_stand
        self.plot_centers = plot_centers

        self.flash_trees = False
        self.current_plot_index = 0
        self.remaining_plots = list(range(len(plot_stand.plots)))
        self.completed_plots = []
        self.current_plot = plot_stand.plots[self.current_plot_index]
        self.stand_center = deepcopy(plot_stand.center)
        self.scale_factor = get_viewport_scale(plot_stand, (800, 600))
        self.plot_transformations = {}
        self.drawing_polygon = False
        self.polygon_points = []
        self.new_plots = []
        self.translate_step = TRANSLATE_STEP
        self.pan_step = PAN_STEP
        self.zoom_step = ZOOM_STEP
        self.tree_scale = TREE_SCALE_INITIAL
        self.event_queue = queue.Queue()
        self.create_ui()
        self.after_id = None
        self.run_pygame()

    def setup_pynput_listener(self):
        self.on_enter = threading.Event()
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

    def on_press(self, key):
        try:
            key_val = key.char
        except AttributeError:
            key_val = key.name
        self.event_queue.put(('press', key_val))

    def on_release(self, key):
        try:
            key_val = key.char
        except AttributeError:
            key_val = key.name
        self.event_queue.put(('release', key_val))

    def process_queue(self):
        try:
            while True:
                event_type, key = self.event_queue.get_nowait()
                if event_type == 'press':
                    self.handle_keydown(key)
                elif event_type == 'release':
                    self.handle_keyup(key)
        except queue.Empty:
            pass
        self.after_id = self.root.after(10, self.process_queue)

    def create_ui(self):
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        button_frame = tk.Frame(control_frame)
        button_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Label(button_frame, text="Camera:").grid(row=0, column=0, padx=5, pady=5)
        tk.Button(button_frame, text="Pan Up", command=lambda: self.pan('up')).grid(row=0, column=1)
        tk.Button(button_frame, text="Pan Left", command=lambda: self.pan('left')).grid(row=0, column=2)
        tk.Button(button_frame, text="Pan Right", command=lambda: self.pan('right')).grid(row=0, column=3)
        tk.Button(button_frame, text="Pan Down", command=lambda: self.pan('down')).grid(row=0, column=4)
        tk.Button(button_frame, text="Zoom In", command=lambda: self.zoom('in')).grid(row=0, column=5)
        tk.Button(button_frame, text="Zoom Out", command=lambda: self.zoom('out')).grid(row=0, column=6)
        tk.Button(button_frame, text="Flash", command=self.toggle_flash).grid(row=0, column=7)

        tk.Label(button_frame, text="Plot:").grid(row=0, column=8, padx=5, pady=5)
        tk.Button(button_frame, text="Shift Up", command=lambda: self.shift_plot('up')).grid(row=0, column=9)
        tk.Button(button_frame, text="Shift Left", command=lambda: self.shift_plot('left')).grid(row=0, column=10)
        tk.Button(button_frame, text="Shift Right", command=lambda: self.shift_plot('right')).grid(row=0, column=11)
        tk.Button(button_frame, text="Shift Down", command=lambda: self.shift_plot('down')).grid(row=0, column=12)
        tk.Button(button_frame, text="Rotate Left", command=lambda: self.rotate_plot('left')).grid(row=0, column=13)
        tk.Button(button_frame, text="Rotate Right", command=lambda: self.rotate_plot('right')).grid(row=0, column=14)
        tk.Button(button_frame, text="Flip", command=self.flip_plot).grid(row=0, column=15)
        tk.Button(button_frame, text="Join", command=self.join_plot).grid(row=0, column=16)
        tk.Button(button_frame, text="Ignore", command=self.ignore_plot).grid(row=0, column=17)
        tk.Button(button_frame, text="Remove", command=self.remove_plot).grid(row=0, column=18)
        tk.Button(button_frame, text="Confirm Plot", command=self.confirm_plot).grid(row=0, column=19)
        tk.Button(button_frame, text="Reset Plot Position", command=self.reset_plot_position).grid(row=0, column=20)
        tk.Button(button_frame, text="Step Back", command=self.step_back).grid(row=0, column=21)
        tk.Button(button_frame, text="New Plot from Polygon", command=self.new_plot_from_polygon).grid(row=0, column=22)
        tk.Button(button_frame, text="Flash Trees", command=self.toggle_flash).grid(row=0, column=23)

        self.pygame_frame = tk.Frame(self.root, width=800, height=600)
        self.pygame_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        list_frame = tk.Frame(self.root)
        list_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        remaining_frame = tk.LabelFrame(list_frame, text="Remaining Plots", padx=5, pady=5)
        remaining_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.remaining_listbox = tk.Listbox(remaining_frame)
        self.remaining_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        for plot_index in self.remaining_plots:
            self.remaining_listbox.insert(tk.END, self.plot_stand.plots[plot_index].plotid)
        self.remaining_listbox.bind('<<ListboxSelect>>', self.on_remaining_plot_select)

        completed_frame = tk.LabelFrame(list_frame, text="Completed Plots", padx=5, pady=5)
        completed_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.completed_listbox = tk.Listbox(completed_frame)
        self.completed_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.after(10, self.process_queue)

    def update_listboxes(self):
        self.remaining_listbox.delete(0, tk.END)
        for plot_index in self.remaining_plots:
            plotid = self.plot_stand.plots[plot_index].plotid
            self.remaining_listbox.insert(tk.END, plotid)
        current_plotid = self.plot_stand.plots[self.current_plot_index].plotid
        if current_plotid in self.remaining_listbox.get(0, tk.END):
            idx = self.remaining_listbox.get(0, tk.END).index(current_plotid)
            self.remaining_listbox.selection_set(idx)
            self.remaining_listbox.see(idx)
        self.completed_listbox.delete(0, tk.END)
        for plot_index in self.completed_plots:
            plotid = self.plot_stand.plots[plot_index].plotid
            self.completed_listbox.insert(tk.END, plotid)

    def on_remaining_plot_select(self, event):
        selection = event.widget.curselection()
        if selection:
            selected_plotid = event.widget.get(selection[0])
            for i, plot in enumerate(self.plot_stand.plots):
                if plot.plotid == selected_plotid:
                    self.current_plot_index = i
                    self.current_plot = self.plot_stand.plots[self.current_plot_index]
                    break
            self.update_listboxes()

    def run_pygame(self):
        os.environ['SDL_WINDOWID'] = str(self.pygame_frame.winfo_id())
        os.environ['SDL_VIDEODRIVER'] = 'windib'
        pygame.init()
        self.screen_size = (800, 600)
        self.screen = pygame.display.set_mode(self.screen_size, pygame.RESIZABLE | pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.running = True
        self.setup_pynput_listener()
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.on_closing()
                elif event.type == pygame.VIDEORESIZE:
                    self.screen_size = event.size
                    self.screen = pygame.display.set_mode(self.screen_size, pygame.RESIZABLE)
                    self.scale_factor = get_viewport_scale(self.plot_stand, self.screen_size)
            self.screen.fill((255, 255, 255))
            if self.chm_stand.trees and not self.flash_trees:
                draw_chm(self.chm_stand.trees, self.screen, self.tree_scale, 1, self.stand_center, self.scale_factor, self.screen_size, tree_component=True)
            elif self.chm_stand.trees and self.flash_trees:
                draw_chm(self.chm_stand.alltrees, self.screen, self.tree_scale, 1, self.stand_center, self.scale_factor, self.screen_size, tree_component=True)
                if self.plot_centers is not None:
                    self.plot_centers.draw_centers(self.screen, (255, 0, 0), 1, self.stand_center, self.scale_factor, self.screen_size)
            if self.current_plot:
                draw_plot(self.screen, self.tree_scale, self.current_plot, 1, self.stand_center, self.scale_factor, self.screen_size, tree_component=True)
            if self.drawing_polygon:
                draw_polygon(self.screen, self.polygon_points)
            pygame.display.flip()
            self.root.update()

    def handle_keyup(self, key):
        if key == 'n' or key == '.':
            self.ignore_plot()
        elif key == 'b':
            self.step_back()
        elif key == 'c':
            self.confirm_plot()
        elif key == 'o':
            self.reset_plot_position()
        elif key == 'f':
            if self.current_plot:
                self.current_plot.coordinate_flip()
        elif key == 'j':
            self.join_plot()
        elif key == 'space':
            self.toggle_flash()
        elif key == 'p':
            self.drawing_polygon = not self.drawing_polygon
            self.polygon_points = []
        elif key == 'd':
            self.remove_plot()

    def handle_keydown(self, key):
        if key in ['left', 'right', 'up', 'down']:
            self.shift_plot(key)
        elif key in ['w', 'a', 's', 'd']:
            self.pan(key)
        elif key == '1':
            self.zoom('in')
        elif key == '2':
            self.zoom('out')
        elif key == '6':
            self.tree_scale *= 1.1
        elif key == '7':
            self.tree_scale *= 0.9
        elif key == '8':
            self.tree_scale = TREE_SCALE_INITIAL
        elif key == 'r':
            self.rotate_plot('left')
        elif key == 'e':
            self.rotate_plot('right')

    def pan(self, direction):
        if direction in ['w', 'up']:
            self.stand_center = (self.stand_center[0], self.stand_center[1] + self.pan_step / self.scale_factor)
        elif direction in ['s', 'down']:
            self.stand_center = (self.stand_center[0], self.stand_center[1] - self.pan_step / self.scale_factor)
        elif direction in ['a', 'left']:
            self.stand_center = (self.stand_center[0] + self.pan_step / self.scale_factor, self.stand_center[1])
        elif direction in ['d', 'right']:
            self.stand_center = (self.stand_center[0] - self.pan_step / self.scale_factor, self.stand_center[1])

    def zoom(self, direction):
        if direction == 'in':
            self.scale_factor *= (1 + self.zoom_step)
        elif direction == 'out':
            self.scale_factor = max(0.01, (1 - self.zoom_step) * self.scale_factor)

    def toggle_flash(self):
        self.flash_trees = not self.flash_trees

    def shift_plot(self, direction):
        if direction == 'up':
            if self.current_plot:
                self.current_plot.translate_plot((0, -self.translate_step))
        elif direction == 'down':
            if self.current_plot:
                self.current_plot.translate_plot((0, self.translate_step))
        elif direction == 'left':
            if self.current_plot:
                self.current_plot.translate_plot((-self.translate_step, 0))
        elif direction == 'right':
            if self.current_plot:
                self.current_plot.translate_plot((self.translate_step, 0))

    def rotate_plot(self, direction):
        if direction == 'left':
            if self.current_plot:
                self.current_plot.rotate_plot(5)
        elif direction == 'right':
            if self.current_plot:
                self.current_plot.rotate_plot(-5)

    def flip_plot(self):
        if self.current_plot:
            self.current_plot.coordinate_flip()

    def join_plot(self):
        if self.current_plot:
            source_array = self.current_plot.get_tree_current_array()[:, -3:]
            target_array = np.array([[tree.x, tree.y, tree.height] for tree in self.chm_stand.trees])
            icp = FractionalICP(source_array, target_array)
            icp.run()
            new_coords = icp.source[:, :2]
            self.current_plot.update_tree_positions(new_coords)

    def ignore_plot(self):
        self.current_plot_index = (self.current_plot_index + 1) % len(self.remaining_plots)
        self.current_plot = self.plot_stand.plots[self.remaining_plots[self.current_plot_index]]
        self.update_listboxes()

    def remove_plot(self):
        if self.current_plot in self.new_plots:
            for tree in self.current_plot.trees:
                tree.original_plot.append_tree(tree)
            self.plot_stand.plots.remove(self.current_plot)
            self.new_plots.remove(self.current_plot)
            if self.completed_plots:
                self.current_plot_index = self.completed_plots.pop()
                self.current_plot = self.plot_stand.plots[self.current_plot_index]
            else:
                self.current_plot_index = self.remaining_plots.pop(0)
                self.current_plot = self.plot_stand.plots[self.current_plot_index]
        self.update_listboxes()

    def confirm_plot(self):
        self.store_transformations(self.current_plot)
        self.completed_plots.append(self.remaining_plots.pop(self.current_plot_index))
        self.chm_stand.remove_matches(self.current_plot, min_dist_percent=15)
        if len(self.remaining_plots) > 0:
            self.current_plot_index = 0
            self.current_plot = self.plot_stand.plots[self.remaining_plots[self.current_plot_index]]
        else:
            self.on_closing()
        self.update_listboxes()

    def store_transformations(self, plot, fail=False):
        R, t, flip = plot.get_transform()
        if not fail:
            self.plot_transformations[plot.plotid] = {
                "original_center": plot.center,
                "final_center": plot.current_center,
                "translation": t,
                "flip": flip,
                "rotation": R
            }
        else:
            self.plot_transformations[plot.plotid] = {
                "original_center": plot.center,
                "final_center": pd.NA,
                "translation": pd.NA,
                "flip": pd.NA,
                "rotation": pd.NA
            }

    def reset_plot_position(self):
        if self.current_plot:
            self.current_plot.reset_transformations()

    def step_back(self):
        if self.completed_plots:
            last_completed = self.completed_plots.pop()
            self.remaining_plots.insert(0, last_completed)
            self.current_plot_index = 0
            self.current_plot = self.plot_stand.plots[self.remaining_plots[self.current_plot_index]]
            if self.current_plot.plotid in self.plot_transformations:
                self.plot_transformations.pop(self.current_plot.plotid)
            self.chm_stand.restore_matches()
        self.update_listboxes()

    def new_plot_from_polygon(self):
        if self.polygon_points:
            trees_to_move = []
            for plot in self.plot_stand.plots:
                for tree in plot.trees:
                    tree_pos = to_screen_coordinates((tree.currentx, tree.currenty), self.stand_center, self.scale_factor, self.screen_size)
                    if is_point_in_polygon(tree_pos, self.polygon_points):
                        logging.info(f"Tree {tree.tree_id} is inside polygon")
                        tree.original_plot = plot
                        trees_to_move.append((tree, plot))
            if self.plot_stand.plots:
                new_plot_id = max(p.plotid for p in self.plot_stand.plots) + 1
            else:
                new_plot_id = 1
            from trees import Plot  # Import here to avoid circular dependency
            new_plot = Plot(new_plot_id, center=np.mean(np.array(self.polygon_points), axis=0))
            for tree, plot in trees_to_move:
                logging.info(f"Removing Tree {tree.tree_id} from Plot {plot.plotid}")
                plot.trees.remove(tree)
                new_plot.append_tree(tree)
            self.plot_stand.add_plot(new_plot)
            logging.info(f"Added new Plot with ID {new_plot.plotid}")
            self.new_plots.append(new_plot)
            self.remaining_plots.append(self.current_plot_index)
            self.current_plot_index = len(self.plot_stand.plots) - 1
            self.current_plot = new_plot
            self.polygon_points = []
            self.update_listboxes()

    def on_closing(self):
        if self.after_id:
            self.root.after_cancel(self.after_id)

        self.running = False
        df_transform = pd.DataFrame.from_dict(self.plot_transformations, orient='index')
        transformation_dir = './Transformations'
        if not os.path.isdir(transformation_dir):
            os.mkdir(transformation_dir)
        df_transform.to_csv(f'{transformation_dir}/Stand_{self.plot_stand.standid}_transformation.csv', index=False)
        if isinstance(self.plot_stand, SavedStand):
            self.plot_stand.write_out().to_csv(f'{self.plot_stand.fp}', index=False)
        else:
            tree_dir = './Trees'
            if not os.path.isdir(tree_dir):
                os.mkdir(tree_dir)
            self.plot_stand.write_out().to_csv(f'{tree_dir}/Stand_{self.plot_stand.standid}_trees.csv', index=False)
        pygame.quit()
        self.root.quit()
        self.root.destroy()


if __name__ == "__main__":
    if len(sys.argv) == 5:
        if int(sys.argv[4]) == 1:
            my_data = SavedStand(ID=sys.argv[1], file_path=sys.argv[2])
            my_chm = CHMPlot(file_path=sys.argv[3], x=my_data.center[0], y=my_data.center[1], dist=70)
            my_plot_centers = None
        elif int(sys.argv[4]) == 2:
            my_data = SavedStand(ID=sys.argv[1], file_path=sys.argv[2])
            my_chm = SavedPlot(file_path=sys.argv[3], x=my_data.center[0], y=my_data.center[1], dist=70)
            my_plot_centers = None
    else:
        my_data = Stand(ID=sys.argv[1], file_path=sys.argv[2])
        my_chm = CHMPlot(file_path=sys.argv[3], x=my_data.center[0], y=my_data.center[1], dist=70)
        my_plot_centers = PlotCenters(sys.argv[2], my_data)
    root = tk.Tk()
    app = App(root, my_data, my_chm, my_plot_centers)
    root.mainloop()
