from PyQt5.QtWidgets import QApplication
from view import loadUi_example
from model import Camera

if __name__ == '__main__':

    camera = Camera(0)
    app = QApplication([])
    start_window = loadUi_example(camera) #loadUi_example is inheriting Camera class
    start_window.show()
    app.exit(app.exec_())  # QApplication and this line are for creating infinite loop
    # In this case, the infinite loop is given by app.exec_(). If you remove that line, you will see that the program runs,
    # but nothing actually happens. . It is important to note that before defining any windows, you should always define the
    # application in which they are going to run.



