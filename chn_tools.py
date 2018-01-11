import Tkinter
import tkFileDialog
import tkMessageBox
import numpy
import struct
import time
import string
import re

class Chn :
    def __init__(self,filename) :
        try:
            fh = open(filename,mode="rb")
        except IOError:
            raise IOError
        except:
            print "Unknown error trying to open %s" % (filename)
            raise IOError
        try:
            buffer = fh.read(32)
        except IOError:
            raise IOError
        except:
            print "Unknown error trying to read %s" % (filename)
            raise IOError
        if 32 != len(buffer):
            print "Attempted to read 32 bytes from %s, only got %d bytes" % (filename,len(buffer))
            raise IOError
        [type, unitnumber, segment_number] = struct.unpack("hhh",buffer[0:6])
        if -1 != type:
            print "File %s has type = %d, it should be -1." % (filename,type)
            raise IOError
        ascii_seconds = buffer[6:8]
        [real_time_20ms, live_time_20ms] = struct.unpack("ii",buffer[8:16])
        start_date = buffer[16:24]
        start_time = buffer[24:28]
        [channel_offset, channel_count] = struct.unpack("hh",buffer[28:32])
        if 0 == 1:
            print "type: %02x" % (type)
            print "unitnumber: %d" % (unitnumber)
            print "segment_number: %d" % (segment_number)
            print "ascii_seconds: %s" % (ascii_seconds)
            print "real time: %d ms, live time %d ms" % (20*real_time_20ms, 20*live_time_20ms)
            print "start date: %s" % (start_date)
            print "start time: %s" % (start_time)
            print "channel offset: %d" % (channel_offset)
            print "channel_count: %d" % (channel_count)
        buf_len = 4 * channel_count
        try:
            buffer = fh.read(buf_len)
        except IOError:
            raise IOError
        except:
            print "Unknown error trying to read %d bytes from %s" % (buf_len,filename)
            raise IOError
        if buf_len != len(buffer):
            print "Attempted to read %d bytes from %s, only got %d bytes" % (buf_len,filename,len(buffer))
            raise IOError            
        fh.close()
        self.spectrum = numpy.zeros(channel_count,dtype=int)
        for i in range(channel_count):
            offset = 4 * i
            [self.spectrum[i]] = struct.unpack("I",buffer[offset:offset+4])
        if "1" == start_date[7]:
            century = 20
        else:
            century = 19
        start_RFC2822 = "%s %s %02d%s %s:%s:%s" % (start_date[0:2], start_date[2:5], century, start_date[5:7], start_time[0:2], start_time[2:4], ascii_seconds)
        #print start_RFC2822
        self.start_time = time.strptime(start_RFC2822,"%d %b %Y %H:%M:%S")
        self.real_time = 0.02*real_time_20ms
        self.live_time = 0.02*live_time_20ms
        #print self.start_time
    def mid_time(self):
        return time.mktime(self.start_time)+0.5*self.real_time
    def activity(self):
        return self.spectrum/self.live_time
    
class Simulation:
    def __init__(self):
        self.isotopes =  [ "212Pb",   "212Bi",  "212Po", "208Tl", "208Pb" ]
        self.colors =    [ "red",     "green",  "blue",  "yellow", "orange" ]
        self.halflives = [10.64*3600, 60.60*60,  0.3e-6,  3*60,   numpy.inf ]
        self.halflives = numpy.array(self.halflives)
        self.dt = 3 # seconds
        self.probabilities = 0.5 / self.halflives * self.dt # probability of decay during dt
        self.probabilities[2] = 1 # half-life is << dt, assume 100% prob of decay
        self.quantities = [  1,         0,         0,      0,        0]
        self.count = len(self.isotopes)
        self.time = 0
    def iterate(self):
        decays = self.probabilities * self.quantities
        self.quantities = self.quantities - decays
        self.quantities[1] = self.quantities[1] + decays[0] # all 212Pb -> 212Bi
        self.quantities[2] = self.quantities[2] + 0.64*decays[1] # 64% 212Bi -> 212Po
        self.quantities[3] = self.quantities[3] + 0.36*decays[1] # 36% 212Pb -> 212Tl
        self.quantities[4] = self.quantities[4] + decays[2] + decays[3]
        self.time = self.time + self.dt
class Alpha:
    def __init__(self):
        self.simulation = Simulation()
        self.step = 0
        self.data = []
        self.mode = 0
        self.time_target = 5*3600
        self.root = Tkinter.Tk()
        self.canvas = Tkinter.Canvas(self.root,bg="black", height=256, width=1024)
        self.x_axis = Tkinter.Canvas(self.root,bg="white", height=32, width=1024)
        buttons = Tkinter.Frame(self.root,relief=Tkinter.RAISED,borderwidth=1)
        # display summary of files loaded
        l_lines_label = Tkinter.Label(buttons,text="Currently loaded:")
        self.lines_var = Tkinter.StringVar()
        self.lines_var.set("None")
        l_lines_value = Tkinter.Label(buttons,textvariable=self.lines_var)
        l_lines_value.pack(side=Tkinter.RIGHT)
        l_lines_label.pack(side=Tkinter.RIGHT)
        b = Tkinter.Button(buttons,text="Load",command=self.callback_load)
        b.pack(side=Tkinter.RIGHT)
        b = Tkinter.Button(buttons,text="Sum",command=self.callback_sum)
        b.pack(side=Tkinter.RIGHT)
        b = Tkinter.Button(buttons,text="Activity",command=self.callback_activity)
        b.pack(side=Tkinter.RIGHT)
        b = Tkinter.Button(buttons,text="Zoom Out",command=self.callback_zoom)
        b.pack(side=Tkinter.RIGHT)
        b = Tkinter.Button(buttons,text="Save",command=self.callback_save)
        b.pack(side=Tkinter.RIGHT)
        b = Tkinter.Button(buttons,text="Quit",command=self.callback_quit)
        b.pack(side=Tkinter.RIGHT)
        self.canvas.pack(side=Tkinter.TOP)
        self.x_axis.pack(side=Tkinter.TOP)
        buttons.pack(side=Tkinter.BOTTOM)
        self.root.after_idle(self.polling)
        self.canvas.bind("<Button>",self.clicked)
        self.canvas.bind("<ButtonRelease>",self.released)
        self.root.mainloop()
    def clicked(self,event):
        self.x_click = event.x            
    def released(self,event):
        if event.x < self.x_click: return
        if 1 != self.mode: return
        w = self.canvas.winfo_width()
        channels = self.hi - self.lo
        lo = self.lo + self.x_click * channels / w
        hi = self.lo + event.x * channels / w
        self.lo = int(numpy.floor(lo))
        self.hi = int(numpy.ceil(hi))
        self.pending = 1
    def polling(self):
        if 0 == self.mode:
            self.animate()
        elif 1 == self.pending:
            self.pending = 0
            if 1 == self.mode:
                self.update_sum()
            elif 2 == self.mode:
                self.update_activity()
        self.root.after(100,self.polling)
    def callback_quit(self):
        self.root.quit()
    def callback_save(self) :
        if 1 == self.mode: return self.save_sum()
        if 2 == self.mode: return self.save_activity()
        tkMessageBox.showwarning("Save file","What are you doing, Dave?\nI can't let you do that, Dave.")
    def save_sum(self):
        data = self.sum_data[self.lo:self.hi]
        channels = len(data)
        options = {}
        options["defaultextension"] = ".data"
        options["filetypes"] = [("Data files",".data"),("All files",".*")]
        options["title"] = "Save summed data to disk"
        options["initialfile"] = "%s-sum-%d-%d" % (self.basename,self.lo,self.hi-1)
        fh = tkFileDialog.asksaveasfile(mode="w",**options)
        if None == fh: return
        for i in range(channels):
            str = "%d\t%d\t%f\n" % (i+self.lo,data[i],numpy.sqrt(data[i]))
            fh.write(str)
        fh.close()
    def save_activity(self):
        options = {}
        options["defaultextension"] = ".data"
        options["filetypes"] = [("Data files",".data"),("All files",".*")]
        options["title"] = "Save activity data to disk"
        options["initialfile"] = "%s-activity-%d-%d" % (self.basename,self.lo,self.hi-1)
        fh = tkFileDialog.asksaveasfile(mode="w",**options)
        if None == fh: return
        for i in range(len(self.activity_x)):
            str = "%f\t%f\t%f\n" % (self.activity_x[i],self.activity_y[i],self.activity_e[i])
            fh.write(str)
        fh.close()
    def callback_zoom(self):
        if 1 != self.mode: return
        self.lo = 0
        self.hi = len(self.sum_data)
        self.pending = 1
    def callback_sum(self):
        self.mode = 1
        count = len(self.data)
        if 0 == count: return
        self.canvas.delete("all")
        spectrum = self.data[0].spectrum
        for i in range(1,count):
            spectrum = spectrum + self.data[i].spectrum
        self.sum_data = spectrum
        self.lo = 0
        self.hi = len(spectrum)
        self.pending = 1
    def callback_activity(self):
        self.mode = 2
        count = len(self.data)
        if 0 == count: return
        x = numpy.zeros(count)
        y = numpy.zeros(count)
        e = numpy.zeros(count)
        for i in range(count):
            x[i] = self.data[i].mid_time()
            data = self.data[i].spectrum[self.lo:self.hi]
            sum = data.sum()
            y[i] = sum / self.data[i].live_time
            e[i] = numpy.sqrt(sum) / self.data[i].live_time
        self.activity_x = x - x.min()
        self.activity_y = y
        self.activity_e = e
        self.pending = 1
    def update_sum(self):
        self.canvas.delete("all")
        data = self.sum_data[self.lo:self.hi]
        v_norm = 1./data.max()
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        channels = self.hi - self.lo
        h_norm = 1./channels*w
        for i in range(channels):
            x1 = int(i*h_norm)
            x2 = int((i+1)*h_norm)
            y = int((h-1)*(1-data[i]*v_norm))
            if h_norm > 3:
                self.canvas.create_rectangle(x1,h-1,x2,y,fill="red")
            else:
                self.canvas.create_line(x1,h-1,x1,y,fill="red")
        self.x_axis.delete("all")
        if channels < 10:
            step = 1
        elif channels < 30:
            step = 2
        elif channels < 60:
            step = 5
        elif channels < 120:
            step = 10
        elif channels < 300:
            step = 20
        elif channels < 600:
            step = 50
        elif channels < 1200:
            step = 100
        elif channels < 3000:
            step = 200
        else:
            step = 500
        start = int(numpy.ceil(self.lo/step))
        stop = int(numpy.floor(self.hi/step))
        for i in range(start,stop):
            x = int(h_norm*(i*step-self.lo))
            self.x_axis.create_text(x,16,text="%.0f" % (i*step))
    def update_activity(self):
        self.canvas.delete("all")
        self.x_axis.delete("all")
        hi = self.activity_y + self.activity_e
        lo = self.activity_y - self.activity_e
        v_norm = 1./hi.max()
        h_norm = 1./self.activity_x.max()
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        for i in range(len(hi)):
            if self.activity_y[i] > 0:
                x = int(h_norm*self.activity_x[i]*w)
                y1 = int(h-v_norm*lo[i]*h)
                y2 = int(h-v_norm*hi[i]*h)
                self.canvas.create_line(x,y1,x,y2,fill="red")
        hours = self.activity_x.max() / 3600.
        if hours > 12:
            step = 2
        else:
            step = 1
        for i in range(1,int(hours/step)):
            x = int(i*step/hours*w)
            self.x_axis.create_text(x,16,text="%d h"%(i*step))
    def callback_load(self):
        files = tkFileDialog.askopenfilenames(filetypes=[("Mastro Spectrum",".Chn")],
                                              title="Select files to load (hint Ctrl-A)")
        if None == files: return
        if 0 == len(files): return
        files = list(files)
        files.sort()
        data = []
        total_time = 0.0
        basename = files[0]
        if basename.count("/") > 0:
            i = string.rindex(basename,"/")+1
            basename = basename[i:]
        if basename.count("\\") > 0:
            i = string.rindex(basename,"\\")+1
            basename = basename[i:]
        chunks = re.split("[0-9]+\.[cC][hH][nN]",basename)
        if len(chunks) > 0:
            basename = chunks[0]
        self.basename = basename
        for i in range(len(files)):
            if i > 0:
                self.canvas.delete(t)
            t = self.canvas.create_text(32,32,
                                        text="Reading %s" % (files[i]),
                                        fill="white",
                                        anchor=Tkinter.W)
            current = Chn(files[i])
            data.append(current)
            total_time = total_time + current.real_time
        self.canvas.delete(t)
        if total_time < 300:
            tstr = "%.1f seconds" % (total_time)
        elif total_time < 7200:
            tstr = "%.1f minutes" % (total_time / 60)
        elif total_time < 2e5:
            tstr = "%.1f hours" % (total_time / 3600)
        else:
            tstr = "%.1f days"    
        self.lines_var.set("%d files, %s" % (len(data),tstr))
        self.data = data
    def animate(self):
        step = self.step + 1
        w = self.canvas.winfo_width()
        if step >= w: 
            print "step = %d, w = %d" % (step,w)
            return
        count = self.simulation.count
        if 1 == step:
            for i in range(count):
                self.canvas.create_text(w-32,32*(i+1),
                                        text=self.simulation.isotopes[i],
                                        fill=self.simulation.colors[i],
                                        anchor=Tkinter.E);

        h = self.canvas.winfo_height()
        before = self.simulation.quantities
        for i in range(100): self.simulation.iterate()
        after = self.simulation.quantities
        for i in range(count):
            x1 = step-1
            y1 = h*(1-before[i])
            x2 = step
            y2 = h*(1-after[i])
            self.canvas.create_line(x1,y1,x2,y2,fill=self.simulation.colors[i],width=2)
        if self.simulation.time > self.time_target:
            self.x_axis.create_text(step,16,text="%.0f h" % (self.simulation.time/3600.0))
            self.time_target = self.time_target + 5*3600
        self.step = step
        
tkFileDialog.Directory()
x = Alpha()