from PyQt5.QtWidgets import QApplication
from view import loadUi_example
from model import Camera

if __name__ == '__main__':

    camera = Camera(0)
    app = QApplication([])
    start_window = loadUi_example(camera)
    start_window.show()
    app.exit(app.exec_())
