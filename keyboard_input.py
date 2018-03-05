KEYS = [ord('0') + i for i in range(1, 4)]

class KeyboardInput:
    def __init__(self):
        self.horizontal = 0
        self.vertical = 0
        self.key = None
        self.finished = False

    def key_press(self, key, mod):
        if key in KEYS:
            self.key = key
        if key == ord('f'):
            self.finished = True

    def key_release(self, key, mod):
        if key == self.key:
            self.key = None

    def add_to_window(self, window):
        window.on_key_press = self.key_press
        window.on_key_release = self.key_release

    def get_action(self):
        if self.finished:
            raise StopIteration
        if self.key == ord('1'):
            return 1
        if self.key == ord('2'):
            return 2
        if self.key == ord('3'):
            return 3
        return 0

    def get_action_loop(self):
        while True:
            if self.finished:
                return
            yield self.get_action()
