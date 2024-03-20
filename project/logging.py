import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as PDF


class Log:
    def __init__(self):
        pass

    def on_training_start(self):
        pass

    def on_training_stop(self):
        pass

    def on_epoch_complete(self, epoch, stats):
        pass


class LogCompose(Log):
    def __init__(self, list_log):
        self.list_log = list_log

    def on_training_start(self):
        for l in self.list_log:
            l.on_training_start()

    def on_training_stop(self):
        for l in self.list_log:
            l.on_training_stop()

    def on_epoch_complete(self, epoch, stats):
        for l in self.list_log:
            l.on_epoch_complete(epoch, stats)


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


class ReportCompiler(Log):
    def __init__(self, filepath, source_filepath):
        self.filepath = filepath
        self.source_filepath = source_filepath

    def on_training_stop(self):
        # Read all out CSV stats
        data = pd.read_csv(self.source_filepath)
        report = PDF.PdfPages(self.filepath)

        # Iterate over all stats
        for name, values in data.items():
            if (name != "epoch"):
                figure = self.get_line_plot(name, values)
                figure.savefig(report, format="pdf")
                plt.close()

        report.close()


    def get_line_plot(self, name, series):
        figure = plt.figure()
        plt.title(f"Graph of {name}")
        plt.xlabel("Epoch")
        plt.plot(series)
        return figure





