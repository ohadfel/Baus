function [saved] = split2trainAndTestSet( X,cond1,cond2,trainTestRatio,yLabled )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

%% split data to train and tset set for Cond1 and Cond2 conditions
saved=0;



[ XTrainCond1, XTestCond1,trialsNumAndLength1,patternsCrossValInd1] = split2trainAndTestSet1Cond( X,cond1 ,yLabled,trainTestRatio);
[ XTrainCond2, XTestCond2,trialsNumAndLength2,patternsCrossValInd2] = split2trainAndTestSet1Cond( X,cond2 ,yLabled,trainTestRatio);

XTrain=[XTrainCond1;XTrainCond2];
YTrain=[ones(size(XTrainCond1,1),1)*0;ones(size(XTrainCond2,1),1)*1];

%trainShuffle=randperm(length(YTrain));
%XTrainShuffled=XTrain(trainShuffle,:);
%YTrainShuffled=YTrain(trainShuffle);



XTest=[XTestCond1;XTestCond2];
YTest=[ones(size(XTestCond1,1),1)*0;ones(size(XTestCond2,1),1)*1];

lastOfCond1=max(patternsCrossValInd1(:,1));
newPatternsCrossValInd2=patternsCrossValInd2;
newPatternsCrossValInd2=newPatternsCrossValInd2(:,1)+lastOfCond1;
patternsCrossValInd=[patternsCrossValInd1;newPatternsCrossValInd2];
save('Xtrain.mat','XTrain','cond1','cond2','-v7.3');
save('Xtest.mat','XTest','cond1','cond2','-v7.3');
save('Ytrain.mat','YTrain','cond1','cond2','-v7.3');
save('Ytest.mat','YTest','cond1','cond2','-v7.3');
save('trials.mat','trialsNumAndLength1','trialsNumAndLength2','-v7.3');
save('inds.mat','patternsCrossValInd','patternsCrossValInd1','patternsCrossValInd2','-v7.3');

saved=1;
end

