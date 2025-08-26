import os
import sys
import queue
import threading
import tkinter as tk
from tkinter import messagebox
import pygame
import logging
import platform
import subprocess
from pynput import keyboard
import pandas as pd
import numpy as np
from copy import deepcopy
import time
import random
# Import custom modules
from trees import Stand, SavedStand
from chm_plot import CHMPlot, SavedPlot
from render import PlotCenters, to_screen_coordinates, get_viewport_scale, draw_plot, draw_chm, draw_polygon, is_point_in_polygon
from ficp import FractionalICP

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def open_in_finder(folder_path):
    """Open the given folder in the system file explorer."""
    if platform.system() == "Windows":
        os.startfile(folder_path)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", folder_path])
    else:  # Assume Linux or similar
        subprocess.Popen(["xdg-open", folder_path])


# Constants
TRANSLATE_STEP = 0.5
PAN_STEP = 5
ZOOM_STEP = 0.3
TREE_SCALE_INITIAL = 1.0

class App:
    """Main application integrating a Tkinter control panel and a Pygame viewport.

    Attributes:
        root (tk.Tk | tk.Toplevel): Parent Tk window for controls and widgets.
        plot_stand (Stand): Layer 1 stand with plots to be registered.
        chm_stand (Plot): Layer 2 stems (CHM or saved data as a single Plot-like object).
        plot_centers (PlotCenters | None): Optional helper for drawing plot centers.
        display_mode (int): 0=all layer2, 1=unmatched layer2, 2=end result (both).
        current_plot_index (int): Index into plot_stand.plots for the active plot.
        remaining_plot_ids (list[int]): Queue of plot IDs yet to be confirmed.
        completed_plot_ids (list[int]): Stack of plot IDs already confirmed.
        stand_center (tuple[float, float]): View center in world units.
        scale_factor (float): Pixels per world unit for the viewport.
        output_folder (str): Optional user-selected folder for tree CSV outputs.
    """
    def __init__(self, root, plot_stand, chm_stand, plot_centers, startup_root=None, output_folder=None):
        self.root = root
        self.window_active = True  # assume active initially
        self.startup_root = startup_root  # To restore reference to startup menu.
        # Default to ./Output if none provided (CLI path, tests, etc.)
        self.output_folder = output_folder or os.path.join(os.getcwd(), "Output")
        os.makedirs(self.output_folder, exist_ok=True)
        # Bind both tkinter focus events (in case the overall app loses focus)
        self.root.bind("<FocusIn>", self.on_focus_in)
        self.root.bind("<FocusOut>", self.on_focus_out)
        
        self.plot_stand = plot_stand
        self.chm_stand = chm_stand
        self.plot_centers = plot_centers
        self.last_space_press = None
        self.display_mode = 0
        # Display Mode
        # 0 = show all trees 
        # 1 = show only unmatched trees
        # 2 = show end result (both layers together).
        self.current_plot_index = 0
        self.remaining_plot_ids = [p.plotid for p in plot_stand.plots]
        self.completed_plot_ids = []
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
        self.listener = None  # Will hold the pynput listener instance.
        self.flash_text = None  # For flashing text messages
        self.flash_end_time = 0
        self.show_help = False
        self.keymap_down, self.keymap_up = self.build_keymaps()
        self.create_ui()
        self.after_id = None
        self.run_pygame()

    def update_caption(self):
        """Set the Pygame window title with Stand/Plot context."""
        try:
            plotid = self.current_plot.plotid if self.current_plot else "None"
            pygame.display.set_caption(
                f"Co-Registration Game — Stand {self.plot_stand.standid} — Plot {plotid}"
            )
        except Exception:
            pygame.display.set_caption("Co-Registration Game")

    def start_listener(self):
        """Start the pynput keyboard listener if not already running."""
        if self.listener is None:
            self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
            self.listener.start()
            logging.info("Keyboard listener started.")

    def stop_listener(self):
        """Stop the pynput keyboard listener if running."""
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
            logging.info("Keyboard listener stopped.")

    def flash_message(self, message, duration=1.5):
        """Set a flash message to be drawn for a given duration (in seconds)."""
        self.flash_text = message
        self.flash_end_time = time.time() + duration

    def _plot_by_id(self, plotid):
        """Return (index, plot) for the given PlotID or (None, None) if not found."""
        for i, p in enumerate(self.plot_stand.plots):
            if p.plotid == plotid:
                return i, p
        return None, None

    def _rebuild_id_queues(self):
        """Rebuild remaining and completed plot ID queues after plot mutations."""
        existing_ids = [p.plotid for p in self.plot_stand.plots if len(p.trees) > 0]
        self.completed_plot_ids = [pid for pid in self.completed_plot_ids if pid in existing_ids]
        self.remaining_plot_ids = [pid for pid in existing_ids if pid not in self.completed_plot_ids]
        # Update current_plot_index to match current_plot's position
        if self.current_plot:
            idx, _ = self._plot_by_id(self.current_plot.plotid)
            self.current_plot_index = idx if idx is not None else 0

    # These tkinter focus handlers now also control the listener.
    def on_focus_in(self, event):
        self.window_active = True
        self.start_listener()
        self.flash_message("MOUSEOVER\nFOCUS GAINED")
        logging.info("Focus gained (tkinter): flashing message and starting listener.")

    def on_focus_out(self, event):
        self.window_active = False
        self.stop_listener()
        self.flash_message("MOUSEOVER \nFOCUS LOST")
        logging.info("Focus lost (tkinter): flashing message and stopping listener.")

    def on_press(self, key):
        if not self.window_active:
            return  # Ignore if the window is not active.
        try:
            key_val = key.char
        except AttributeError:
            key_val = key.name
        self.event_queue.put(('press', key_val.lower() if isinstance(key_val, str) else key_val))

    def on_release(self, key):
        if not self.window_active:
            return  # Ignore if the window is not active.
        try:
            key_val = key.char
        except AttributeError:
            key_val = key.name
        self.event_queue.put(('release', key_val.lower() if isinstance(key_val, str) else key_val))

    def process_queue(self):
        # --- Start Change ---
        # Check if the root window (Toplevel) associated with this App instance still exists.
        # If not, don't process the queue and don't reschedule.
        if not self.root or not self.root.winfo_exists():
             logging.debug("process_queue: Root window destroyed, stopping queue processing.")
             self.after_id = None # Ensure it's cleared
             return # Stop processing
        # --- End Change ---

        try:
            while True:
                event_type, key = self.event_queue.get_nowait()
                # Check window existence again before handling, just in case.
                if not self.root or not self.root.winfo_exists(): break

                if event_type == 'press':
                    self.handle_keydown(key)
                elif event_type == 'release':
                    self.handle_keyup(key)
        except queue.Empty:
            pass
        except tk.TclError as e:
             # Catch potential errors during key handling if window is destroyed mid-process
             logging.warning(f"process_queue: TclError during event handling (window likely destroyed): {e}")
             self.after_id = None
             return # Stop processing

        # --- Start Change ---
        # Only reschedule if the window still exists
        if self.root and self.root.winfo_exists():
            self.after_id = self.root.after(10, self.process_queue)
        else:
             logging.debug("process_queue: Root window destroyed, not rescheduling.")
             self.after_id = None
        # --- End Change ---

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
        for plotid in self.remaining_plot_ids:
            self.remaining_listbox.insert(tk.END, plotid)
        self.remaining_listbox.bind('<<ListboxSelect>>', self.on_remaining_plot_select)

        completed_frame = tk.LabelFrame(list_frame, text="Completed Plots", padx=5, pady=5)
        completed_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.completed_listbox = tk.Listbox(completed_frame)
        self.completed_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.after(10, self.process_queue)

    
    def update_listboxes(self):
        # --- Start Change ---
        # Check if the listbox widgets still exist before trying to modify them
        if not self.remaining_listbox or not self.remaining_listbox.winfo_exists():
            logging.warning("update_listboxes: Remaining listbox does not exist. Aborting update.")
            return
        if not self.completed_listbox or not self.completed_listbox.winfo_exists():
            logging.warning("update_listboxes: Completed listbox does not exist. Aborting update.")
            return
        # --- End Change ---

        try:
            self.remaining_listbox.delete(0, tk.END)
            for plotid in self.remaining_plot_ids:
                # Check listbox existence again before inserting
                if not self.remaining_listbox.winfo_exists():
                    break
                self.remaining_listbox.insert(tk.END, str(plotid))

            # Check existence before proceeding
            if not self.remaining_listbox.winfo_exists(): return

            current_plotid = str(self.current_plot.plotid)
            listbox_items = self.remaining_listbox.get(0, tk.END)
            if current_plotid in listbox_items:
                 idx = listbox_items.index(current_plotid)
                 # Check existence before selection/see
                 if self.remaining_listbox.winfo_exists():
                     self.remaining_listbox.selection_clear(0, tk.END) # Clear selection first
                     self.remaining_listbox.selection_set(idx)
                     self.remaining_listbox.see(idx)

            # Check existence before deleting/inserting into completed listbox
            if not self.completed_listbox.winfo_exists(): return

            self.completed_listbox.delete(0, tk.END)
            for plotid in self.completed_plot_ids:
                if not self.completed_listbox.winfo_exists():
                    break
                self.completed_listbox.insert(tk.END, str(plotid))

        except tk.TclError as e:
            logging.error(f"Error updating listboxes (likely window destroyed): {e}")



    def on_remaining_plot_select(self, event):
        selection = event.widget.curselection()
        if selection:
            selected_plotid = int(event.widget.get(selection[0]))
            idx, plot = self._plot_by_id(selected_plotid)
            if plot is not None:
                self.current_plot_index = idx
                self.current_plot = plot
        self.update_listboxes()

    def run_pygame(self):
        # Only set windib on Windows for compatibility on other platforms.
        if platform.system() == "Windows":
            os.environ.setdefault('SDL_VIDEODRIVER', 'windib')
        pygame.init()
        # Create a font for flash messages.
        self.font = pygame.font.SysFont(None, 36)
        self.screen_size = (800, 600)
        # Use SCALED | RESIZABLE for smoother resizing in windowed mode.
        self.screen = pygame.display.set_mode(
            self.screen_size, pygame.SCALED | pygame.RESIZABLE
        )
        self.update_caption()

        self.root.lower() #Send tk window to back

        self.running = True
        # Start the listener initially.
        self.start_listener()
        clock = pygame.time.Clock()
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.on_closing()
                elif event.type == pygame.VIDEORESIZE:
                    self.screen_size = event.size
                    self.screen = pygame.display.set_mode(
                        self.screen_size, pygame.SCALED | pygame.RESIZABLE
                    )
                    self.scale_factor = get_viewport_scale(self.plot_stand, self.screen_size)
                # Prefer modern Pygame 2 window focus events when available
                elif hasattr(pygame, "WINDOWEVENT") and event.type == pygame.WINDOWEVENT:
                    if event.event == pygame.WINDOWEVENT_FOCUS_LOST and self.window_active:
                        self.window_active = False
                        self.stop_listener()
                        self.flash_message("MOUSEOVER \nFOCUS LOST")
                        logging.info("Pygame window focus lost: stopped keyboard listener.")
                    elif event.event == pygame.WINDOWEVENT_FOCUS_GAINED and not self.window_active:
                        self.window_active = True
                        self.start_listener()
                        self.flash_message("MOUSEOVER\nFOCUS GAINED")
                        logging.info("Pygame window focus gained: restarted keyboard listener.")
                # Handle focus events from the pygame window.
                elif event.type == pygame.ACTIVEEVENT:
                    # Check if the keyboard focus state changed (state 1)
                    if event.state & 1:
                        if event.gain == 0 and self.window_active:
                            self.window_active = False
                            self.stop_listener()
                            self.flash_message("MOUSEOVER \nFOCUS LOST")
                            logging.info("Pygame window lost focus: stopped keyboard listener.")
                        elif event.gain == 1 and not self.window_active:
                            self.window_active = True
                            self.start_listener()
                            self.flash_message("MOUSEOVER\nFOCUS GAINED")
                            logging.info("Pygame window gained focus: restarted keyboard listener.")
            self.screen.fill((255, 255, 255))
            # Draw CHM and plots based on display_mode.
            if self.chm_stand.trees: 
                if self.display_mode == 0:
                    # Draw all trees.
                    draw_chm(stems=self.chm_stand.alltrees, screen=self.screen, tree_scale=self.tree_scale, 
                             alpha=1, stand_center=self.stand_center, scale_factor=self.scale_factor, screen_size=self.screen_size, tree_component=True)
                elif self.display_mode == 1:
                    # Draw only unmatched trees.
                    draw_chm(stems=self.chm_stand.trees, screen=self.screen, tree_scale=self.tree_scale, 
                             alpha=1, stand_center=self.stand_center, scale_factor=self.scale_factor, screen_size=self.screen_size, tree_component=True)
                elif self.display_mode == 2:
                    # Draw end result: both matched and unmatched trees.
                    draw_chm(stems=self.chm_stand.alltrees, screen=self.screen, tree_scale=self.tree_scale, 
                             alpha=1, stand_center=self.stand_center, scale_factor=self.scale_factor, screen_size=self.screen_size, tree_component=True)
                    for i, plot in enumerate(self.plot_stand.plots):
                        draw_plot(screen=self.screen, tree_scale=self.tree_scale, plot=plot, alpha=1,
                                  stand_center=self.stand_center, scale_factor=self.scale_factor, 
                                  screen_size=self.screen_size, tree_component=True, fill_color=(0,255,0))

            if self.current_plot:
                draw_plot(self.screen, self.tree_scale, self.current_plot, 1, self.stand_center, self.scale_factor, self.screen_size, tree_component=True)
            if self.drawing_polygon:
                draw_polygon(self.screen, self.polygon_points)

            #Draw flash message if active
            self.draw_flash_message()
            if self.show_help:
                self.draw_help_overlay()

            pygame.display.flip()
            try:
                self.root.update()
            except tk.TclError:
                # Tk window destroyed; stop gracefully
                self.running = False
            # Cap frame rate to reduce CPU load
            clock.tick(60)

    def draw_flash_message(self):
        """Draws the flash message on the pygame screen if it is still active."""
        if self.flash_text and time.time() < self.flash_end_time:
            # Split the message into lines.
            lines = self.flash_text.split('\n')
            total_height = 0
            surfaces = []
            for line in lines:
                text_surface = self.font.render(line, True, (255, 0, 0))
                surfaces.append(text_surface)
                total_height += text_surface.get_height()
            # Center the text on the screen.
            y = (self.screen_size[1] - total_height) // 2
            for surface in surfaces:
                x = (self.screen_size[0] - surface.get_width()) // 2
                self.screen.blit(surface, (x, y))
                y += surface.get_height()
        elif self.flash_text:
            # Clear the flash message when the duration has expired.
            self.flash_text = None


    def draw_help_overlay(self):
        lines = ["Shortcuts:"] + [f"{k}: {v}" for (k, v) in self.help_entries]
        surf_list = [self.font.render(line, True, (0, 0, 0)) for line in lines]
        pad = 8
        width = max(s.get_width() for s in surf_list) + 2 * pad
        height = sum(s.get_height() for s in surf_list) + 2 * pad
        x = self.screen_size[0] - width - 10
        y = 10
        bg = pygame.Surface((width, height), pygame.SRCALPHA)
        bg.fill((255, 255, 255, 200))
        self.screen.blit(bg, (x, y))
        ty = y + pad
        for s in surf_list:
            self.screen.blit(s, (x + pad, ty))
            ty += s.get_height()

    def handle_keyup(self, key):
        fn = self.keymap_up.get(key)
        if fn:
            fn()

    def handle_keydown(self, key):
        fn = self.keymap_down.get(key)
        if fn:
            fn()

    def build_keymaps(self):
        """Create dictionaries for keydown and keyup handlers + display help text."""
        kd = {
            'left':  lambda: self.shift_plot('left'),
            'right': lambda: self.shift_plot('right'),
            'up':    lambda: self.shift_plot('up'),
            'down':  lambda: self.shift_plot('down'),
            'w':     lambda: self.pan('up'),
            'a':     lambda: self.pan('left'),
            's':     lambda: self.pan('down'),
            'd':     lambda: self.pan('right'),
            '1':     lambda: self.zoom('in'),
            '2':     lambda: self.zoom('out'),
            '6':     lambda: setattr(self, 'tree_scale', self.tree_scale * 1.1),
            '7':     lambda: setattr(self, 'tree_scale', self.tree_scale * 0.9),
            '8':     lambda: setattr(self, 'tree_scale', TREE_SCALE_INITIAL),
            'e':     lambda: self.rotate_plot('left'),
            'r':     lambda: self.rotate_plot('right'),
            'h':     self.toggle_help,
        }
        ku = {
            'n':     self.ignore_plot,
            '.':     self.mark_unplaceable,
            'b':     self.step_back,
            'c':     self.confirm_plot,
            'o':     self.reset_plot_position,
            'f':     self.flip_plot,
            'j':     self.join_plot,
            'p':     self.toggle_polygon_mode,
            'x':     self.remove_plot,
            'space': self.handle_space,
        }
        self.help_entries = [
            ("W/A/S/D", "Pan"),
            ("Arrow Keys", "Shift plot"),
            ("1 / 2", "Zoom in / out"),
            ("6 / 7 / 8", "Tree scale up / down / reset"),
            ("E / R", "Rotate CCW / CW"),
            ("F", "Flip plot vertically"),
            ("J", "Join (Fractional ICP)"),
            ("C", "Confirm plot"),
            ("N", "Skip plot"),
            (".", "Mark unplaceable"),
            ("B", "Step back"),
            ("X", "Remove plot"),
            ("O", "Reset plot position"),
            ("P", "Polygon split mode"),
            ("Space", "Toggle unmatched/all (double-tap: end result)"),
            ("H", "Toggle help overlay"),
        ]
        return kd, ku

    def toggle_polygon_mode(self):
        self.drawing_polygon = not self.drawing_polygon
        self.polygon_points = []

    def handle_space(self):
        now = time.time()
        if self.last_space_press and (now - self.last_space_press < 0.3):
            self.display_mode = 0 if self.display_mode == 2 else 2
            self.last_space_press = None
        else:
            self.last_space_press = now
            self.root.after(300, self.toggle_flash)

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
        # If last_space_press is still set, then no second press was received.
        if self.last_space_press:
            # Toggle between display modes 0 and 1.
            if self.display_mode == 0:
                self.display_mode = 1
            elif self.display_mode == 1:
                self.display_mode = 0
            self.last_space_press = None

    def toggle_help(self):
        self.show_help = not self.show_help
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
        if self.current_plot is None or not self.current_plot.trees:
            self.flash_message("No trees in current plot")
            return
        if not self.chm_stand.trees:
            self.flash_message("No CHM trees to match against")
            return
        source_array = self.current_plot.get_tree_current_array()[:, -3:]
        target_array = np.array([[tree.x, tree.y, tree.height] for tree in self.chm_stand.trees])
        icp = FractionalICP(source_array, target_array)
        icp.run()
        new_coords = icp.source[:, :2]
        self.current_plot.update_tree_positions(new_coords)

    def ignore_plot(self):
        """Skip to the next remaining plot without altering queues or writing transforms."""
        if not self.remaining_plot_ids:
            return
        current_id = self.current_plot.plotid
        if current_id in self.remaining_plot_ids:
            pos = self.remaining_plot_ids.index(current_id)
            next_plot_id = self.remaining_plot_ids[(pos + 1) % len(self.remaining_plot_ids)]
        else:
            # If current plot isn't in remaining (e.g., already completed), go to the first remaining.
            next_plot_id = self.remaining_plot_ids[0]
        idx, plot = self._plot_by_id(next_plot_id)
        if plot is not None:
            self.current_plot_index = idx
            self.current_plot = plot
        self.update_listboxes()

    def mark_unplaceable(self):
        self.store_transformations(self.current_plot, fail=True)
        if self.current_plot.plotid in self.remaining_plot_ids:
            idx_in_queue = self.remaining_plot_ids.index(self.current_plot.plotid)
            self.completed_plot_ids.append(self.remaining_plot_ids.pop(idx_in_queue))
        if self.remaining_plot_ids:
            next_id = self.remaining_plot_ids[0]
            idx, plot = self._plot_by_id(next_id)
            if plot is not None:
                self.current_plot_index = idx
                self.current_plot = plot
        else:
            # If no plots remain, save results and exit workflow.
            self.save_files()
            self.show_success_dialog()
        self.update_listboxes()

    def remove_plot(self):
        if self.current_plot in self.new_plots:
            for tree in self.current_plot.trees:
                tree.original_plot.append_tree(tree)
            self.plot_stand.plots.remove(self.current_plot)
            self.new_plots.remove(self.current_plot)
            self._rebuild_id_queues()
            if self.completed_plot_ids:
                last_id = self.completed_plot_ids.pop()
                idx, plot = self._plot_by_id(last_id)
                if plot is not None:
                    self.current_plot_index = idx
                    self.current_plot = plot
            elif self.remaining_plot_ids:
                next_id = self.remaining_plot_ids[0]
                idx, plot = self._plot_by_id(next_id)
                if plot is not None:
                    self.current_plot_index = idx
                    self.current_plot = plot
            else:
                self.current_plot_index = 0
                self.current_plot = None
        self.update_listboxes()

    def confirm_plot(self):
        # Check if we're on the last plot and confirm before final saving.
        if len(self.remaining_plot_ids) == 1:
            confirm_save = messagebox.askyesno("Confirm Save",
                                            "You have confirmed the last plot.\n\n"
                                            "Are you sure you want to save the results to files?")
            if not confirm_save:
                return
            else:
                self.store_transformations(self.current_plot)
                if self.current_plot.plotid in self.remaining_plot_ids:
                    idx_in_queue = self.remaining_plot_ids.index(self.current_plot.plotid)
                    self.completed_plot_ids.append(self.remaining_plot_ids.pop(idx_in_queue))
                self.chm_stand.remove_matches(self.current_plot, min_dist_percent=15)
                self.save_files()
                self.show_success_dialog()
        else:
            self.store_transformations(self.current_plot)
            if self.current_plot.plotid in self.remaining_plot_ids:
                idx_in_queue = self.remaining_plot_ids.index(self.current_plot.plotid)
                self.completed_plot_ids.append(self.remaining_plot_ids.pop(idx_in_queue))
            self.chm_stand.remove_matches(self.current_plot, min_dist_percent=15)
            if self.remaining_plot_ids:
                # Set the current plot to the first ID in the remaining queue.
                next_id = self.remaining_plot_ids[0]
                idx, plot = self._plot_by_id(next_id)
                if plot is not None:
                    self.current_plot_index = idx
                    self.current_plot = plot
        self.update_listboxes()

    def save_files(self):
        """Save trees and transformation files."""
        df_transform = pd.DataFrame.from_dict(self.plot_transformations, orient='index')
        # Preserve Plot IDs as a column in the CSV.
        df_transform.index.name = 'PlotID'
        df_transform = df_transform.reset_index()
        transformation_dir = './Transformations'
        if not os.path.isdir(transformation_dir):
            os.mkdir(transformation_dir)
        df_transform.to_csv(
            f'{transformation_dir}/Stand_{self.plot_stand.standid}_transformation.csv',
            index=False,
        )


        # Save tree data to the specified output folder
        if not os.path.isdir(self.output_folder):
            os.makedirs(self.output_folder, exist_ok=True)
        tree_path = os.path.join(self.output_folder, f'Stand_{self.plot_stand.standid}_trees.csv')
        self.plot_stand.write_out().to_csv(tree_path, index=False)

    def show_success_dialog(self):
        """Show a dialog after a successful save with options to show files, continue, or exit."""
        logging.info("Showing success dialog.")
        dialog = tk.Toplevel(self.root)
        dialog.title("Successfully saved!")

        # Make the dialog modal
        dialog.grab_set()
        dialog.transient(self.root)

        # --- Define Button Actions ---

        def do_show_files():
            """Opens the output folders without closing the dialog."""
            logging.info("Show Files button clicked.")
            output_folder_trees = os.path.abspath(self.output_folder)
            output_folder_trans = os.path.abspath('./Transformations')
            logging.info(f"Opening folders: {output_folder_trees}, {output_folder_trans}")
            try:
                # Ensure open_in_finder is defined/imported if used
                open_in_finder(output_folder_trees)
                open_in_finder(output_folder_trans)
            except Exception as e:
                logging.error(f"Error opening folders: {e}")
                messagebox.showerror("Error", f"Could not open folders:\n{e}", parent=dialog)

        def do_continue():
            """Closes this dialog and calls the main on_closing handler
            to return to the startup menu."""
            logging.info("Continue button clicked.")
            try:
                dialog.destroy() # Close this dialog first
            except tk.TclError as e:
                logging.warning(f"Error destroying success dialog (Continue): {e}")
            # This should trigger cleanup and show startup menu
            self.on_closing()

        def do_exit():
            """Closes this dialog, cleans up, and terminates the entire application gracefully."""
            logging.info("Exit button clicked. Terminating application.")
            try:
                # Ensure the dialog variable exists and the window exists before destroying
                if 'dialog' in locals() and isinstance(dialog, tk.Toplevel) and dialog.winfo_exists():
                    dialog.destroy()
            except tk.TclError as e:
                logging.warning(f"Error destroying success dialog (Exit): {e}")

            # Cancel the pending process_queue call
            if self.after_id:
                try:
                    # Check if the Toplevel window (self.root) still exists before cancelling
                    if self.root and self.root.winfo_exists():
                        self.root.after_cancel(self.after_id)
                        logging.info("Pending 'after' call cancelled (Exit).")
                    else:
                        logging.info("App Toplevel window already gone, cannot cancel 'after' (Exit).")
                except tk.TclError as e:
                    logging.warning(f"Error cancelling 'after' during exit: {e}")
                self.after_id = None

            self.running = False # Stop pygame loop variable if checked elsewhere

            # Attempt to quit Pygame
            try:
                pygame.quit()
                logging.info("Pygame quit successfully (Exit).")
            except pygame.error as e:
                logging.warning(f"Pygame quit error during exit: {e}")

            # Attempt to destroy the main app window (Toplevel)
            try:
                if self.root and self.root.winfo_exists():
                    self.root.destroy()
                    logging.info("App Toplevel window destroyed (Exit).")
            except tk.TclError as e:
                logging.warning(f"Error destroying App Toplevel window during exit: {e}")

            # --- Start Change ---
            # Attempt to gracefully QUIT the main Tkinter loop via the startup_root.
            # DO NOT destroy startup_root here - let the mainloop handle it upon exit.
            try:
                if self.startup_root and self.startup_root.winfo_exists():
                    logging.info("Quitting Tkinter main loop via startup_root...")
                    self.startup_root.quit() # Signal mainloop in startup.py to exit cleanly
                else:
                    # If startup_root is already gone for some reason, we must force exit
                    logging.warning("Startup root invalid or destroyed before quit signal. Forcing exit.")
                    sys.exit()
            except tk.TclError as e:
                 # Catch error if quit fails unexpectedly
                 logging.warning(f"Error quitting main loop via startup_root: {e}. Forcing exit.")
                 sys.exit()

        # --- Configure Dialog Close Button ---
        # Make the 'X' button behave like the Exit button
        dialog.protocol("WM_DELETE_WINDOW", do_exit)

        # --- Create and Layout Widgets ---

        tk.Label(dialog, text="Files saved successfully!").pack(padx=20, pady=(20, 10))

        frame_row1 = tk.Frame(dialog)
        frame_row1.pack(pady=(5, 5))
        show_files_button = tk.Button(frame_row1, text="Show Files", command=do_show_files)
        show_files_button.pack()

        frame_row2 = tk.Frame(dialog)
        frame_row2.pack(pady=(5, 20))

        continue_button = tk.Button(frame_row2, text="Continue", command=do_continue)
        continue_button.pack(side=tk.LEFT, padx=10)

        exit_button = tk.Button(frame_row2, text="Exit", command=do_exit)
        exit_button.pack(side=tk.LEFT, padx=10)

        # Center the dialog
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (dialog.winfo_width() // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

        dialog.wait_window()
        # This point might not be reached if Exit is clicked, as sys.exit() is called.
        logging.info("Success dialog finished (likely via Continue).")


    def store_transformations(self, plot, fail=False):
        """Store per-plot transformation details for later CSV export."""
        if not plot.trees:
            # If the plot is empty, store NA values for all primitives.
            self.plot_transformations[plot.plotid] = {
                "original_center": tuple(map(float, plot.center)),
                "final_center": pd.NA,
                "tx": pd.NA,
                "ty": pd.NA,
                "r00": pd.NA,
                "r01": pd.NA,
                "r10": pd.NA,
                "r11": pd.NA,
                "flip": pd.NA,
            }
            return
        R, t, flip = plot.get_transform()
        if not fail:
            self.plot_transformations[plot.plotid] = {
                "original_center": tuple(map(float, plot.center)),
                "final_center": tuple(map(float, plot.current_center)),
                "tx": float(t[0]),
                "ty": float(t[1]),
                "r00": float(R[0, 0]),
                "r01": float(R[0, 1]),
                "r10": float(R[1, 0]),
                "r11": float(R[1, 1]),
                "flip": bool(flip),
            }
        else:
            self.plot_transformations[plot.plotid] = {
                "original_center": tuple(map(float, plot.center)),
                "final_center": pd.NA,
                "tx": pd.NA,
                "ty": pd.NA,
                "r00": pd.NA,
                "r01": pd.NA,
                "r10": pd.NA,
                "r11": pd.NA,
                "flip": pd.NA,
            }


    def reset_plot_position(self):
        if self.current_plot:
            self.current_plot.reset_transformations()

    def step_back(self):
        """Restore the last confirmed plot back to the remaining queue and make it current."""
        if self.completed_plot_ids:
            last_completed = self.completed_plot_ids.pop()
            self.remaining_plot_ids.insert(0, last_completed)
            idx, plot = self._plot_by_id(last_completed)
            if plot is not None:
                self.current_plot_index = idx
                self.current_plot = plot
            if self.current_plot.plotid in self.plot_transformations:
                self.plot_transformations.pop(self.current_plot.plotid)
            self.chm_stand.restore_matches()
        self.update_listboxes()

    def new_plot_from_polygon(self):
        PolygonDrawingWindow(self)

    def on_polygon_confirmed(self, polygon_points):
        if not polygon_points:
            return

        trees_to_move = []
        affected_plots = {}  # dictionary to count selected trees per plot

        # Iterate over all plots and collect trees that fall inside the polygon.
        for plot in self.plot_stand.plots:
            for tree in plot.trees[:]:
                if is_point_in_polygon((tree.currentx, tree.currenty), polygon_points):
                    trees_to_move.append((tree, plot))
                    affected_plots[plot] = affected_plots.get(plot, 0) + 1

        if not trees_to_move:
            return #Exit without making any changes if no tree points are affected.

        # If exactly one plot is affected and all its trees are selected, do nothing.
        if len(affected_plots) == 1:
            for plot, count in affected_plots.items():
                if count == len(plot.trees):
                    logging.info(f"All trees in Plot {plot.plotid} were selected. No split will be performed.")
                    return  # Exit without making any changes.

        # Otherwise, proceed with splitting the trees into a new plot.
        # Determine the new plot ID.
        new_plot_id = max(p.plotid for p in self.plot_stand.plots) + 1 if self.plot_stand.plots else 1
        from trees import Plot  # Avoid circular dependency.
        new_center = np.mean(np.array(polygon_points), axis=0)
        new_plot = Plot(new_plot_id, center=new_center)

        # Remove selected trees from their original plots and add them to the new plot.
        for tree, plot in trees_to_move:
            logging.info(f"Removing Tree {tree.tree_id} from Plot {plot.plotid}")
            # Remember original plot and preserve current coordinates across append/reset.
            tree.original_plot = plot
            cx, cy = tree.currentx, tree.currenty
            if tree in plot.trees:
                plot.trees.remove(tree)
            new_plot.append_tree(tree)
            tree.currentx, tree.currenty = cx, cy

        # Record transformations for affected plots.
        for plot in set([p for (_, p) in trees_to_move]):
            self.store_transformations(plot)

        # Remove any plots that have become empty.
        self.plot_stand.plots = [p for p in self.plot_stand.plots if len(p.trees) > 0]

        # Add the new plot to the stand.
        self.plot_stand.add_plot(new_plot)
        logging.info(f"Added new Plot with ID {new_plot.plotid}")
        self.new_plots.append(new_plot)

        # Rebuild the plot ID queues based on the updated plot list.
        self._rebuild_id_queues()
        # Set the new plot as the current plot.
        idx, plot = self._plot_by_id(new_plot.plotid)
        if plot is not None:
            self.current_plot_index = idx
            self.current_plot = plot
        self.update_listboxes()
    
    
    def on_closing(self):
        logging.info("Closing App window...")
        self.running = False # Stop the pygame loop
    
        # Stop keyboard listener cleanly
        if self.listener:
            self.stop_listener()
            logging.info("Keyboard listener stopped.")
    
        # Cancel any pending Tkinter 'after' jobs for this window
        if self.after_id:
             try:
                 self.root.after_cancel(self.after_id)
                 logging.info("Pending 'after' call cancelled.")
             except tk.TclError as e:
                  logging.warning(f"Could not cancel 'after' call (window likely closing): {e}")
             self.after_id = None
    
        # Quit Pygame
        try:
            pygame.quit()
            logging.info("Pygame quit successfully.")
        except pygame.error as e:
             logging.warning(f"Pygame quit error (might be already quit or not init): {e}")
    
        # Destroy the Toplevel window associated with the App instance
        try:
            # Ensure self.root still exists before destroying
            if self.root and self.root.winfo_exists():
                 self.root.destroy()
                 logging.info("App Toplevel window destroyed.")
            else:
                 logging.info("App Toplevel window already destroyed or invalid.")
        except tk.TclError as e:
             logging.error(f"Error destroying App Toplevel window: {e}")
    
        # Re-show the startup menu if the reference exists and it's still valid
        if self.startup_root and self.startup_root.winfo_exists():
            try:
                self.startup_root.deiconify() # Show the startup menu again
                # Optional: Bring the startup window to the front
                # bring_window_to_front(self.startup_root)
                logging.info("Startup root deiconified and brought to front.")
            except tk.TclError as e:
                logging.error(f"Error deiconifying startup root (might be destroyed): {e}")
                # If we can't show startup, exit the whole application cleanly
                try:
                     # Attempt to quit the mainloop associated with the startup root
                     if self.startup_root and self.startup_root.winfo_exists():
                         self.startup_root.quit()
                except:
                     pass # Ignore errors if already gone
                sys.exit() # Exit application
        else:
            logging.warning("Startup root reference not found or window destroyed, exiting application.")
            sys.exit() # Exit if no startup menu to return to

#Given an arbitrary integer as plotid
def grey_from_plotid(plotid):
    random.seed(plotid)  # seed with the plot id so it's deterministic
    shade = random.randint(50, 240)  # choose a value between 50 and 240 for visibility
    return f"#{shade:02x}{shade:02x}{shade:02x}"

# Canvas drawing polygon
class PolygonDrawingWindow:
    def __init__(self, parent_app):
        self.parent_app = parent_app
        self.root = tk.Toplevel(parent_app.root)
        self.root.title("Polygon Drawing")
        self.canvas_width = 800
        self.canvas_height = 600
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Initialize viewport parameters using the parent's current values.
        self.center = parent_app.stand_center  # geo coordinates (x,y)
        self.scale = parent_app.scale_factor     # pixels per meter (or similar unit)
        
        self.polygon_points = []  # stored as geo coordinates
        
        # Bind mouse events:
        self.canvas.bind("<Button-1>", self.on_left_click)             # left-click to add a point
        self.canvas.bind("<Button-3>", self.on_right_click)            # right-click to remove the last point
        self.canvas.bind("<Shift-Button-3>", self.on_shift_right_click)  # shift-right-click to remove the closest point
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)          # mouse wheel to zoom
        
        # Confirm and Cancel buttons (and keyboard 'C' for confirm)
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM)
        confirm_btn = tk.Button(btn_frame, text="Confirm (C)", command=self.confirm)
        confirm_btn.pack(side=tk.LEFT, padx=5, pady=5)
        cancel_btn = tk.Button(btn_frame, text="Cancel", command=self.cancel)
        cancel_btn.pack(side=tk.LEFT, padx=5, pady=5)
        self.root.bind("c", lambda event: self.confirm())
        
        self.draw()

    def geo_to_canvas(self, geo_point):
        """Convert geo coordinates to canvas coordinates."""
        canvas_x = (geo_point[0] - self.center[0]) * self.scale + self.canvas_width / 2
        canvas_y = (geo_point[1] - self.center[1]) * self.scale + self.canvas_height / 2
        return (canvas_x, canvas_y)

    def canvas_to_geo(self, x, y):
        """Convert canvas coordinates to geo coordinates."""
        geo_x = (x - self.canvas_width / 2) / self.scale + self.center[0]
        geo_y = (y - self.canvas_height / 2) / self.scale + self.center[1]
        return (geo_x, geo_y)

    def draw(self):
        '''Redraw the canvas contents'''
        self.canvas.delete("all")
        # Draw all Layer 1 trees (from every plot)
        all_plots = self.parent_app.plot_stand.plots
        for plot in all_plots:
            color = grey_from_plotid(plot.plotid)
            for tree in plot.trees:
                pos = self.geo_to_canvas((tree.currentx, tree.currenty))
                # Use the same scaling as in the main window:
                r = max(int(tree.stemdiam * 10 * self.scale / 2), 1) * self.parent_app.tree_scale
                self.canvas.create_oval(pos[0]-r, pos[1]-r, pos[0]+r, pos[1]+r,
                                         fill=color, outline=color)
        # Draw the polygon (if any) on top.
        if len(self.polygon_points) > 1:
            canvas_points = [self.geo_to_canvas(pt) for pt in self.polygon_points]
            flat_points = [coord for point in canvas_points for coord in point]
            self.canvas.create_line(*flat_points, fill="green", width=2)
        # Draw each vertex with a green border.
        for pt in self.polygon_points:
            cx, cy = self.geo_to_canvas(pt)
            r = 5
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                    fill="white", outline="green", width=2)


    def on_left_click(self, event):
        # Left-click: add a new point.
        geo_pt = self.canvas_to_geo(event.x, event.y)
        self.polygon_points.append(geo_pt)
        # Auto-center on the new point.
        #self.center = geo_pt
        self.draw()

    def on_right_click(self, event):
        # Right-click: remove the last point.
        if self.polygon_points:
            self.polygon_points.pop()
            self.draw()

    def on_shift_right_click(self, event):
        # Shift-right-click: remove the point closest to the mouse.
        if not self.polygon_points:
            return
        geo_pt = self.canvas_to_geo(event.x, event.y)
        distances = [((pt[0] - geo_pt[0]) ** 2 + (pt[1] - geo_pt[1]) ** 2) ** 0.5 for pt in self.polygon_points]
        min_index = distances.index(min(distances))
        self.polygon_points.pop(min_index)
        self.draw()

    def on_mouse_wheel(self, event):
        # Adjust zoom factor.
        if event.delta > 0:
            self.scale *= 1.1
        else:
            self.scale /= 1.1
        self.draw()

    def confirm(self):
        # Pass the polygon points back to the main application.
        self.parent_app.on_polygon_confirmed(self.polygon_points)
        self.root.destroy()

    def cancel(self):
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
        # PlotCenters expects only the Stand.
        my_plot_centers = PlotCenters(my_data)
    root = tk.Tk()
    default_out = os.path.join(os.getcwd(), "Output")
    os.makedirs(default_out, exist_ok=True)
    app = App(root, my_data, my_chm, my_plot_centers, output_folder=default_out)
    root.mainloop()
