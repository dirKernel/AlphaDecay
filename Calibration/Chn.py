import struct
import numpy
import time

class Chn :
    def __init__(self,filename) :
        try:
            fh = open(filename,mode="rb")
        except IOError:
            raise IOError
        except:
            print("Unknown error trying to open %s" % (filename))
            raise IOError
        try:
            buffer = fh.read(8736)
        except IOError:
            raise IOError
        except:
            print("Unknown error trying to read %s" % (filename))
            raise IOError
        fh.close()
        if 8736 != len(buffer):
            print("Attempted to read 8736 bytes from %s, only got %d bytes" % (filename,len(buffer)))
            raise IOError
        [type, unitnumber, segment_number] = struct.unpack("hhh",buffer[0:6])
        ascii_seconds = buffer[6:8].decode()
        [real_time_20ms, live_time_20ms] = struct.unpack("ii",buffer[8:16])
        start_date = buffer[16:24].decode()
        start_time = buffer[24:28].decode()
        [channel_offset, channel_count] = struct.unpack("hh",buffer[28:32])
        if 0 == 1:
            print("type: %02x" % (type))
            print("unitnumber: %d" % (unitnumber))
            print("segment_number: %d" % (segment_number))
            print("ascii_seconds: %s" % (ascii_seconds))
            print("real time: %d ms, live time %d ms" % (20*real_time_20ms, 20*live_time_20ms))
            print("start date: %s" % (start_date))
            print("start time: %s" % (start_time))
            print("channel offset: %d" % (channel_offset))
            print("channel_count: %d" % (channel_count))
        self.spectrum = numpy.zeros(channel_count,dtype=int)
        for i in range(channel_count):
            [self.spectrum[i]] = struct.unpack("I",buffer[32+4*i:36+4*i])
        if "1" == start_date[7]:
            century = 20
        else:
            century = 19
        start_RFC2822 = "%s %s %02d%s %s:%s:%s" % (start_date[0:2], start_date[2:5], century, start_date[5:7], start_time[0:2], start_time[2:4], ascii_seconds)
        #print(start_RFC2822)
        self.start_time = time.strptime(start_RFC2822,"%d %b %Y %H:%M:%S")
        self.real_time = 0.02*real_time_20ms
        self.live_time = 0.02*live_time_20ms
        #print(self.start_time)
    def mid_time(self):
        return time.mktime(self.start_time)+0.5*self.real_time
    def activity(self):
        return self.spectrum/self.live_time