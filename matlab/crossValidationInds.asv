function [ patternsCrossValInd ] = crossValidationInds( trialsNumAndLength,numOfFolds )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    [~, IX] =sort(trialsNumAndLength,2);
    sortedTrialsNumAndLength=trialsNumAndLength(:,IX(1,:));
    numOfPatterns=sum(trialsNumAndLength(2,:));
    patternsCrossValInd=zeros(numOfPatterns,3);
    patternsCrossValInd(:,1)=1:numOfPatterns;
        
    counter=1;
     from=1;
     to=sortedTrialsNumAndLength(2,counter);
     
    for ii=sortedTrialsNumAndLength(1,:)
        patternsCrossValInd(from:to,2)=ii;
        from = to+1;
        to = to+sortedTrialsNumAndLength(2,min(counter+1,length(sortedTrialsNumAndLength)));
        counter = counter+1;
    end
   
    delta=floor(length(trialsNumAndLength)/numOfFolds);
    from=1;
    to=from+delta-1;
    for ii=1:numOfFolds
        curVec=trialsNumAndLength(1,from:to);
        for jj=curVec
            patternsCrossValInd(patternsCrossValInd(:,2)==jj,3)=ii;
        end
        from=to+1;
        to=to+delta;
    end
end

