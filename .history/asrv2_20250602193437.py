import importlib
from PyQt5.QtCore import QFileSystemWatcher

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
try:
    import config
except ImportError:
    config = None

class ParticleWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.particles = []
        self.num_particles = PARTICLE_COUNT
        self.current_app_state = "idle"
        self.audio_level_norm = 0.0
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(int(1000 / PARTICLE_ANIMATION_FPS))
        self.elapsed_timer = QElapsedTimer()
        self.elapsed_timer.start()
        self.current_base_orb_radius_px = min(self.width(), self.height()) * PARTICLE_BASE_ORB_RADIUS_RATIO  
        self.reload_config()

    def reload_config(self):
        """Hot reload configuration from config.py if present"""
        global PARTICLE_COUNT
        if config:
            try:
                importlib.reload(config)
                self.num_particles = config.PARTICLE_CONFIG.get('num_particles', PARTICLE_COUNT)
                # You can add more config param reloads here as needed
                PARTICLE_COUNT = self.num_particles
            except Exception as e:
                print(f"Config reload error: {e}")
        self._init_particles()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # ... existing code ...
        self.setup_hot_reload()
        # ... existing code ...

    def setup_hot_reload(self):
        self.file_watcher = QFileSystemWatcher(self)
        self.file_watcher.addPath(__file__)
        if config:
            try:
                import os
                config_path = os.path.abspath(config.__file__)
                self.file_watcher.addPath(config_path)
            except Exception:
                pass
        self.file_watcher.fileChanged.connect(self.on_file_changed)
        self.reload_timer = QTimer()
        self.reload_timer.setSingleShot(True)
        self.reload_timer.timeout.connect(self.reload_particles)

    def on_file_changed(self, path):
        print(f"File changed: {path}")
        self.reload_timer.start(500)

    def reload_particles(self):
        try:
            self.particle_view.reload_config()
            print("Particles reloaded!")
        except Exception as e:
            print(f"Reload error: {e}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F5:
            self.particle_view.reload_config()
            print("Manual reload triggered!")
        super().keyPressEvent(event)
# ... existing code ... 