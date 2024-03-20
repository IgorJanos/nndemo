

class Log:
    def __init__(self):
        pass

    def on_training_start(self):
        pass

    def on_training_stop(self):
        pass

    def on_epoch_complete(self, epoch, stats):
        pass


class CSVLog(Log):
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = None
        self.lines = 0
        self.separator = ","

    def on_training_start(self):
        # Open a file for writing
        self.lines = 0
        self.file = open(str(self.filepath), "w")

    def on_training_stop(self):
        # Close if open
        if (self.file is not None):
            self.file.close()
            self.file = None

    def on_epoch_complete(self, epoch, stats):
        values = { "epoch": epoch }
        values.update(stats)
        if (self.file is not None):
            if self.lines == 0:
                # CSV header
                line = self.separator.join(list(values.keys()))
                self.file.write(line + "\n")

            # Concatenate all values
            line = [ f"{values[k]}" for k in values.keys() ]
            line = self.separator.join(line)

            # Write & Flush IO
            self.file.write(line + "\n")
            self.file.flush()
            self.lines += 1
            





