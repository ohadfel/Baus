function allMarkers = uniteAllOKtimes(markerFile)
% read the marker.mrk file and find all continuous times
%   allMarkers = uniteAllOKtimes(markerFile);
%
% markerFile - full path to the MarkerFile.mrk (including the file name)
%
% allMarkers - struct array.  Each element for one marker with fields
%      Name - char, the name of the marker
%     Times - array with times it happened from start of file

% jun_2013  MA

%% initialize
fid = fopen(markerFile, 'rt');
if fid<1
    error('MATLAB:MEGanalysis:BadFileName',...
        'File %s does not exist!',markerFile)
end
% key phrases
numMark = 'NUMBER OF MARKERS:';
name = 'NAME:';
numSample = 'NUMBER OF SAMPLES:';
trialT = 'TRIAL NUMBER';
% initial definitions
allMarkers = struct('Name',[] , 'Times',[]);
% kk =1;
line=fgetl(fid);

%% search for openning line
while ~strncmpi(line,numMark, length(numMark))
    line = fgetl(fid);
    if line==-1  %EOF
        fclose (fid);
        error('MATLAB:MEGanalysis:BadFile',...
            'File does not contain %s!',numMark)
    end
end
line=fgetl(fid);
nMarkers = sscanf(line, '%d');
allMarkers = repmat(allMarkers, 1,nMarkers);
% allTimes = NaN(1,200*nMarkers);

%% go over all trial types
for mm = 1:nMarkers
    % search for NAME:
    while ~strncmpi(line,name, length(name))
        line = fgetl(fid);
        if line==-1  %EOF
            fclose (fid);
            error('MATLAB:MEGanalysis:BadFile',...
                'After %d searches did not find %s!',mm, name)
        end
    end
    line = fgetl(fid);
    allMarkers(mm).Name = line;
    % search for 'NUMBER...'
    while ~strncmpi(line,numSample, length(numSample))
        line = fgetl(fid);
        if line==-1  %EOF
            fclose (fid);
            error('MATLAB:MEGanalysis:BadFile',...
                'After %d searches did not find %s!',mm, numSample)
        end
    end
    % read the number of Data here
    line=fgetl(fid);
    nTimes = sscanf(line, '%d');
    allTimes = NaN(1,nTimes);
    
    %% collect data from this trial
    while ~strncmpi(line,trialT, length(trialT))
        line = fgetl(fid);
        if line==-1  %EOF
            fclose (fid);
            error('MATLAB:MEGanalysis:BadFile',...
                'After %d searches did not find %s!',mm, trialT)
        end
    end
    for nn = 1:nTimes
        line=fgetl(fid);
        A = sscanf(line, '%f', 2);
        allTimes(nn) = A(2);
    end
    allMarkers(mm).Times = allTimes;
end

%% unite all times
fclose (fid);

return
