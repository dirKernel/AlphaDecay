function [ spectrum ] = get_spectrum( filename )
%get_spectrum Reads spectrum files SPM, SPT and CHN
%   returns a structure with the following fields
[ fd, msg ] = fopen(filename,'r');
if (fd < 0)
  msg = [ msg, ' error returned by attempt to open "', filename, '"' ];
  error('get_spectrum:openError',msg);
end
matches = regexpi(filename,'\.spm');
if (size(matches,2) == 1)
  spectrum = get_spectrum_spm (fd);
else
  matches = regexpi(filename,'\.spt');
  if (size(matches,2) == 1)
    spectrum = get_spectrum_spt (fd);
  else
    matches = regexpi(filename,'\.chn');
    if (size(matches,2) == 1)
      spectrum = get_spectrum_chn (fd);
    else
      error('get_spectrum:identError','does not match extension of known spectrum files (spm,spt,chn)');
    end
  end
end
fclose(fd);
end

function [ spectrum ] = get_spectrum_chn ( fd )
% internal code for chn
type = fread(fd,1,'int16=>int16');
if (type ~= -1)
  error('get_spectrum_chn:headerError','Error reading header ... type != -1')
end
unit_number = fread(fd,1,'uint16=>unit16');
segment_number = fread(fd,1,'uint16=>unit16');
ascii_seconds = fread(fd,2,'uint8=>char')';
real_time_20ms = fread(fd,1,'uint32=>unit32');
live_time_20ms = fread(fd,1,'uint32=>unit32');
start_date = fread(fd,8,'uint8=>char')';
start_time = fread(fd,4,'uint8=>char')';
channel_offset = fread(fd,1,'uint16=>unit16');
spectrum.numbchanspm = fread(fd,1,'uint16=>uint16');
spectrum.data = fread(fd,spectrum.numbchanspm,'uint32=>uint32');
spectrum.pca_date = [ start_date(1,3:5), ' ', start_date(1,1:2), ' ', sprintf('%02d',19+start_date(1,8)-'0'), start_date(1,6:7) ];
spectrum.acqstart_date = spectrum.pca_date;
hours = sscanf(start_time(1,1:2),'%d');
minutes = sscanf(start_time(1,3:4),'%d');
seconds = sscanf(ascii_seconds(1,1:2),'%d');
spectrum.acqstart_time = sprintf('%02d:%02d:%02d %cm',1+mod(hours-1,12),minutes,seconds,'a'+(hours>12).*('p'-'a'));
spectrum.acquire_start = 24*3600*datenum(spectrum.acqstart_date)+3600*hours+60*minutes+seconds;
spectrum.acquire_stop = spectrum.acquire_start + 0.02*real_time_20ms;
spectrum.etime = 0.02*live_time_20ms;
end

function [ spectrum ] = get_spectrum_spm ( fd )
% internal code for SPM
bcd = fread (fd, 3, 'uint8=>uint');
spectrum.etime = 0;
weight = 1;
for i = 1:3
  spectrum.etime = spectrum.etime + weight .* bitand(bcd(i),15);
  weight = 10 .* weight;
  spectrum.etime = spectrum.etime + weight .* bitand(bitshift(bcd(i),4),15);
  weight = 10 .* weight;
end
spectrum.ltimeflag = fread(fd,1,'uint8=>unit8');
spectrum.rtimeflag = fread(fd,1,'uint8=>uint8');
spectrum.convergain = fread(fd,1,'uint8=>uint8');
spectrum.digoffset = fread(fd,1,'uint8=>uint8');
spectrum.idcodestr = fread(fd,15,'uint8=>char')';
spectrum.pca_date = fread(fd,12,'uint8=>char')';
spectrum.pca_time = fread(fd,12,'uint8=>char')';
spectrum.group = fread(fd,1,'uint8=>uint8');
spectrum.units = fread(fd,1,'uint8=>uint8');
spectrum.calib = fread(fd,110,'uint8=>uint8');
spectrum.expansion = fread(fd,99,'uint8=>uint8');
spectrum.phamode = fread(fd,1,'uint8=>uint8');
spectrum.mcs = fread(fd,1,'uint8=>uint8');
spectrum.mcstimelab = fread(fd,1,'uint8=>uint8');
spectrum.mcsdwellnumb = fread(fd,1,'uint8=>uint8');
spectrum.mcspasct = fread(fd,3,'uint8=>uint8');
spectrum.centupflag = fread(fd,1,'uint8=>uint8');
spectrum.fwhmflag = fread(fd,1,'uint8=>uint8');
spectrum.numbchanspm = fread(fd,1,'uint16=>uint16');
spectrum.asday = fread(fd,1,'uint8=>uint8');
spectrum.asmonth = fread(fd,1,'uint8=>uint8');
spectrum.asyear = fread(fd,1,'uint16=>uint16');
spectrum.ashund = fread(fd,1,'uint8=>uint8');
spectrum.assec = fread(fd,1,'uint8=>uint8');
spectrum.asmin = fread(fd,1,'uint8=>uint8');
spectrum.ashour = fread(fd,1,'uint8=>uint8');
spectrum.astpday = fread(fd,1,'uint8=>uint8');
spectrum.astpmonth = fread(fd,1,'uint8=>uint8');
spectrum.astpyear = fread(fd,1,'uint16=>uint16');
spectrum.astphund = fread(fd,1,'uint8=>uint8');
spectrum.astpsec = fread(fd,1,'uint8=>uint8');
spectrum.astpmin = fread(fd,1,'uint8=>uint8');
spectrum.astphour = fread(fd,1,'uint8=>uint8');
spectrum.id = fread(fd,72,'uint8=>char')';
spectrum.majvers = fread(fd,1,'uint8=>uint8');
spectrum.minvers = fread(fd,1,'uint8=>uint8');
spectrum.real_elap_time = fread(fd,1,'uint32=>uint32');
spectrum.acqstop_time = fread(fd,13,'uint8=>char')';
spectrum.acqstop_date = fread(fd,13,'uint8=>char')';
spectrum.acqstart_time = fread(fd,13,'uint8=>char')';
spectrum.acqstart_date = fread(fd,13,'uint8=>char')';
spectrum.future = fread(fd,96,'uint8=>uint8')';
spectrum.endheader = fread(fd,1,'uint16=>uint16')';
if (100*spectrum.majvers + spectrum.minvers ~= spectrum.endheader)
  error('get_spectrum_spm:headerError','Error reading header ... version mismatch')
end
% Caution ahead.
% MATLAB defines datenum as number of days since the epoch, multiplying by
% seconds/day appears to be equivalent to time_t
year = double(spectrum.asyear);
month = double(spectrum.asmonth);
day = double(spectrum.asday);
hour = double(spectrum.ashour);
min = double(spectrum.asmin);
sec = double(spectrum.assec);
hund = double(spectrum.ashund) ./ 100.0;
spectrum.acquire_start = 24*3600*datenum(year,month,day,hour,min,sec)+hund;
year = double(spectrum.astpyear);
month = double(spectrum.astpmonth);
day = double(spectrum.astpday);
hour = double(spectrum.astphour);
min = double(spectrum.astpmin);
sec = double(spectrum.astpsec);
hund = double(spectrum.astphund) ./ 100.0;
spectrum.acquire_stop = 24*3600*datenum(year,month,day,hour,min,sec)+hund;
spectrum.data = fread(fd,spectrum.numbchanspm,'uint32=>uint32');
end
