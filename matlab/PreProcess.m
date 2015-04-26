%% Preprocess
patternMaxLen = 40;
cond1 = 37;
cond2 = 36;
preStimNumOfSamples =1/1000;
postStimNumOfSamples = 392/1000;


hostname = char( getHostName( java.net.InetAddress.getLocalHost ) );
if strcmp(hostname, 'Ohad-PC')
    path='C:\Users\Ohad\Copy\Baus\Code\matlab\';
else
    path='/home/ohadfel/Copy/Baus/Code/matlab/';
end

% Load Idan's data
% load('CCDppBySNRnew.mat');
% load('547pntHull.mat');
% 
% 
% CCD547=CCDpp(indSelectedBalls,:);
% 
% 
% clear PRIvol;
% clear newBalls;
% clear newPosBalls;
% clear CCDpp;
% clear indSelectedBalls;
% 
% Find the samples where a bau was found
% activeSamples=find(sum(CCD547));
% save('activeSamples.mat','activeSamples','samplingRate','-v7.3');
% 
% sampleBetweenActiveSamples = diff(activeSamples);
% 
% cumsumSampleBetweenActiveSamples = cumsum(sampleBetweenActiveSamples);
% cumsumIter=cumsumSampleBetweenActiveSamples;
% 
% patternIndRange=zeros(length(activeSamples),2);
% test=5000;
% 
% tic;
% for ii=1:length(activeSamples)
% for ii=1:test
%     disp(ii);
%     endIndOfPattern=activeSamples(ii)+cumsumIter(max(find(cumsumIter<=patternMaxLen)));
%     if isempty(endIndOfPattern)
%         endIndOfPattern=activeSamples(ii);
%     end
%     patternIndRange(ii,:) = [activeSamples(ii),endIndOfPattern]; 
%     cumsumIter=cumsumIter-cumsumIter(1);
%     cumsumIter=cumsumIter(2:end);
% end
% toc
% 
% activityInNet=sum(CCD547);
% activityInNetP=activityInNet(activeSamples);
% locations=zeros(max(activityInNet),length(activeSamples));
% for ii=1:length(activeSamples)
%     locations(1:sum(CCD547(:,activeSamples(ii))),ii)=find(CCD547(:,activeSamples(ii)));
% end
% 
% create X feature vec
% X=ones(size(patternIndRange,1),547)*-1;
% for patternsIter=1:size(patternIndRange,1)
% for patternsIter=1:2
%     disp(['pattern number - ',num2str(patternsIter)]);
%     flags=zeros(1,547);
%     liveInd4location = find(activeSamples>=patternIndRange(patternsIter,1) & activeSamples<=patternIndRange(patternsIter,2));
%     liveInd4samples = activeSamples(liveInd4location);
%      
%     base=liveInd4samples(1);
% 
%     for jj=1:length(liveInd4location)
%         locationsInSample = locations(:,liveInd4location(jj));
%         locationsInSample(locationsInSample==0) = [];
%         for kk=1:length(locationsInSample)
%             X(patternsIter,locationsInSample(kk))=liveInd4samples(jj)-base;
%             flags(locationsInSample(kk))=flags(locationsInSample(kk))+1;
%         end
%     end
%     if max(flags)>1
%         disp('~~~~~~~~~~~~~~~~~~~~~~~~~~~ERROR~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
%     end
% end
% 
% X=X+1;
% save('newX.mat','X','-v7.3');

%%
load('activeSamples.mat')
allMarkers = uniteAllOKtimes('NewMarker.mrk');
headlines = cell(size(allMarkers));
numOfTimes = zeros(size(allMarkers));

for ii=1:size(allMarkers,2)-2
   headlines{ii} = allMarkers(ii).Name;
   numOfTimes(ii) = length(allMarkers(ii).Times);
   disp(['marker number ',num2str(ii),' min is ',num2str(min(allMarkers(ii).Times)*samplingRate)])
end

allCondTimes = zeros(size(allMarkers,2)-2,max(numOfTimes));
allCondInd = zeros(size(allMarkers,2)-2,max(numOfTimes));
for ii=1:size(allMarkers,2)-2
   allCondTimes(ii,1:numOfTimes(ii))=sort(allMarkers(ii).Times);
   allCondInd(ii,:)=(allCondTimes(ii,:)*samplingRate);
end


% preStimNumOfSamples = -100;
% postStimNumOfSamples = 392;



bottomLimInd=allCondInd + preStimNumOfSamples*samplingRate;
topLimInd=allCondInd + postStimNumOfSamples*samplingRate;

bottomLimInd(bottomLimInd==preStimNumOfSamples)=inf;
topLimInd(topLimInd==preStimNumOfSamples)=inf;

% 
% bottomLimInd=allCondInd ;
% topLimInd=allCondInd ;

y4activeSamples = zeros(size(activeSamples'));
indsFromTrig = zeros(size(activeSamples'));


for ii=1:length(activeSamples)
    [cond,trial] = find(and(bottomLimInd<=activeSamples(ii),topLimInd>activeSamples(ii)));
    if max(size(cond))>1
        
        if(sum(cond==37)==1)
            trial = trial(find(cond==37));
            cond = 37;
            
        elseif(sum(cond==36)==1)
            trial = trial(find(cond==36));
            cond=36;
        else
%             disp('problem');
%             disp('too much lables for samples');
%             disp(cond);
                y4activeSamples(ii)=-1;
        end
    end
    if length(cond)==1
        y4activeSamples(ii) = cond;
        indsFromTrig(ii) = activeSamples(ii)-floor(allCondInd(cond,trial));
        disp(['ii=',num2str(ii),' and cond is ',headlines{cond}]);
    end
end

save('y.mat','y4activeSamples','indsFromTrig','-v7.3');


%%
clear
load('activeSamples.mat');
load('newX.mat');
allMarkers = uniteAllOKtimes('NewMarker.mrk');
headlines = cell(size(allMarkers));
numOfTimes = zeros(size(allMarkers));

%% Get conditions headlines and legth
for ii=1:size(allMarkers,2)-2
   headlines{ii} = allMarkers(ii).Name;
   numOfTimes(ii) = length(allMarkers(ii).Times);
   disp(['marker number ',num2str(ii),' min is ',num2str(min(allMarkers(ii).Times)*samplingRate)])
end

allCondTimes = zeros(size(allMarkers,2)-2,max(numOfTimes));
allCondInd = zeros(size(allMarkers,2)-2,max(numOfTimes));
for ii=1:size(allMarkers,2)-2
   allCondTimes(ii,1:numOfTimes(ii))=sort(allMarkers(ii).Times);
   allCondInd(ii,:)=(allCondTimes(ii,:)*samplingRate);
end

allCondIndAsVec=reshape(allCondInd,1,[]);
allCondIndAsVec=sort(allCondIndAsVec);
allCondIndAsVec(allCondIndAsVec==0)='';

%%
yCorrect=zeros(size(allCondIndAsVec));

for ii=1:length(allCondIndAsVec)
    [cond,~]=find(allCondInd==allCondIndAsVec(ii));
     if(sum(cond==37)==1)
        cond=37;
    elseif(sum(cond==36)==1)
        cond=36;
     elseif (sum(cond==14)+sum(cond==15)>0)
         cond=-1;
     end
    yCorrect(ii)=cond;
end
%save('yCorrect.mat','yCorrect');
save('y4allCondsTrigs.mat','yCorrect','-v7.3');

%%
load('y.mat');
y4activeSamplesWithinCond=y4activeSamples;
y4activeSamplesWithinCond(y4activeSamplesWithinCond==0)='';
yD=diff([0;y4activeSamplesWithinCond]);
switchOfConds=yD;
switchOfConds(yD~=0)=1;
yLabled=switchOfConds.*y4activeSamplesWithinCond;
yN=y4activeSamplesWithinCond(yD~=0);

%%
inds=find(yLabled);
fullMatrix=zeros(length(inds)-1,4);
fullMatrix(:,1)=inds(1:end-1);
fullMatrix(:,2)=inds(2:end)-1;
fullMatrix(:,3)=yLabled(inds(1:end-1));

for ii=1:size(fullMatrix,1)
    fullMatrix(ii,4)=length(find(fullMatrix(1:ii,3)==fullMatrix(ii,3)));
end
%% split data to train and tset set for LAST and Change conditions
cond1 = 37;
cond2 = 36;

trainTestRatio = 0.8;
numOftrialsCond1 = fullMatrix(find(fullMatrix(:,3)==cond1,1,'last'),4);
[ trainIndsCond1,testIndCond1 ] = trainTestSplit( numOftrialsCond1,trainTestRatio );

numOftrialsCond2 = fullMatrix(find(fullMatrix(:,3)==cond2,1,'last'),4);
[ trainIndsCond2,testIndCond2 ] = trainTestSplit( numOftrialsCond2,trainTestRatio );

saved=split2trainAndTestSet(X,cond1,cond2,trainTestRatio,yLabled);
