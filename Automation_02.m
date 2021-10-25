%%
clear all;

%%
for i = 39:39
    image_register(i);
end




















%%
function [] = image_register(numberofimage)

%%
% numberofimage=1;
if (numberofimage<9)
    targetim = strcat('reg_slide_', '0', num2str(numberofimage));
    referenceim = strcat('reg_slide_','0', num2str(numberofimage+1));
end
if (numberofimage==9)
    targetim = strcat('reg_slide_', '0',num2str(numberofimage));
    referenceim = strcat('reg_slide_', num2str(numberofimage+1));
end
if (numberofimage>9)
    targetim = strcat('reg_slide_', num2str(numberofimage));
    referenceim = strcat('reg_slide_', num2str(numberofimage+1));
end

%%
targetimage = strcat(targetim,'.jpg');
referenceimage = strcat(referenceim,'.jpg');

rotation_name = strcat(targetim,'_3500rotate.jpg');
centroid_name = strcat(targetim,'1_3500rotate.jpg');

%%
image19 = (imread(targetimage)); %%target
image20 = (imread(referenceimage)); %%reference
image19 = imcrop(image19,[1 551 3499 3499]);
image20 = imcrop(image20,[1 551 3499 3499]);

target_rgb=mat2gray(image19);
target=rgb2gray(target_rgb);

ref_rgb=mat2gray(image20);
ref=rgb2gray(ref_rgb);

%%
optimizer = registration.optimizer.OnePlusOneEvolutionary;
optimizer.InitialRadius=1e-04;
metric = registration.metric.MattesMutualInformation;
tform = imregtform(target, ref, 'similarity', optimizer, metric);

image19rotate(:,:,1)=imwarp(image19(:,:,1),tform,'OutputView',imref2d(size(ref)));
image19rotate(:,:,2)=imwarp(image19(:,:,2),tform,'OutputView',imref2d(size(ref)));
image19rotate(:,:,3)=imwarp(image19(:,:,3),tform,'OutputView',imref2d(size(ref)));

%%
imshow(image19rotate);
figure
imshow(image20);

%%
imwrite(image19rotate,rotation_name);%%follow target
extract_centroid(rotation_name,40,centroid_name);%%target

%%
image191rotate = imread(centroid_name);
figure
imshow(image191rotate)

%%
c=3500;
r=3500;
cell=[];
for c1 = 1:c
    for r1 = 1:r
        if (image191rotate(r1,c1) > 200)%%记得改
            cell=[cell;[r1,c1]];
        end
    end
end

%%
a=cell(:,1);
b=cell(:,2);
cell(:,1)=b;
cell(:,2)=a;

%%
imshow(image19rotate)%%记得改
hold on
plot(cell(:,1),cell(:,2),'y*')
hold off

%%
cell_default=cell;%%正

%%
clear cell;
cell=cell_default;%%反

%%
%%444
clear cell20 length stdsize;
stdsize = size(cell);
length = stdsize(1,1);
cell20=[];
crecord = [];
for k = 1:length%%改数字
    x=cell(k,1);
    y=cell(k,2);
    image19s = imcrop(image19rotate,[x-100 y-100 200 200]);%%x,y,width,height
    image20s = imcrop(image20,[x-200 y-200 400 400]);
    c = normxcorr2(image19s(:,:,1),image20s(:,:,1));
    [ypeak,xpeak] = find(c==max(c(:)));
    maxc = max(c(:));
    crecord = [crecord;maxc];
    Sz=size(xpeak);
    if (Sz(1,1) ~= 1)
        disp('wrong')
        disp(ypeak)
        disp(xpeak)
        ypeak = ypeak(1,1)
        xpeak = xpeak(1,1)
    end
    yoffSet = ypeak-100;
    xoffSet = xpeak-100;
    cell20=[cell20;[xoffSet,yoffSet]];%%小正方形的中心点   
end

%%
cell_default1=cell;%%正
cell20_default1=cell20;%%正
crecord_default1=crecord;%%正

%%
%%default1
clear cell cell20 crecord cell20_diff realcell20;
cell=cell_default1;%%反
cell20=cell20_default1;%%反
crecord=crecord_default1;%%反
cell20_diff(:,1) = cell20(:,1)-200;%%小正方形的中心点-大正方形的中心点=diff
cell20_diff(:,2) = cell20(:,2)-200;
realcell20=cell+cell20_diff;

%%
%%remove outliner001---有点太粗糙了，需要改进,就是因为NCC方法不好，所以才accuracy这么低
clear index length stdsize;
stdsize = size(cell);
length = stdsize(1,1);
index=[];
for j=1:length%%改数字
    if (cell20(j,1)>50 & cell20(j,1)<350 & cell20(j,2)>50 & cell20(j,2)<350)
        index = [index,j]%%范围左150右150，上150下150
    end 
end
index = unique(index);

cell20 = cell20(index,:);
cell = cell(index,:);
crecord = crecord(index,:);

clear length stdsize;
stdsize = size(cell);
length = stdsize(1,1);

%%
cell_default2=cell;%%正
cell20_default2=cell20;%%正
crecord_default2=crecord;%%正

%%
%%复原default2
clear cell cell20 crecord cell20_diff;
cell=cell_default2;%%反
cell20=cell20_default2;%%反
crecord=crecord_default2;%%反
cell20_diff(:,1) = cell20(:,1)-200;%%小正方形的中心点-大正方形的中心点=diff
cell20_diff(:,2) = cell20(:,2)-200;
realcell20=cell+cell20_diff;

%%
[cell20,cell,realcell20,cell20_diff,crecord,length] = ...
        removeOutliners(2,2,1.1,cell,realcell20,cell20,crecord);

%%
try
    for i = 1:10
    %%without std_plane
    [cell20,cell,realcell20,cell20_diff,crecord,length] = ...
        removeOutliners(2,2,1.1,cell,realcell20,cell20,crecord);
    end
    for i = 1:10
    %%without std_plane
    [cell20,cell,realcell20,cell20_diff,crecord,length] = ...
        removeOutliners(6,6,1.2,cell,realcell20,cell20,crecord);
    end
catch
    disp(targetimage);
end

%%
clear length;
length = size(cell)
length = length(1,1)
if (length<=3)
    
    %%复原default2
    clear cell cell20 crecord cell20_diff;
    cell=cell_default2;%%反
    cell20=cell20_default2;%%反
    crecord=crecord_default2;%%反
    cell20_diff(:,1) = cell20(:,1)-200;%%小正方形的中心点-大正方形的中心点=diff
    cell20_diff(:,2) = cell20(:,2)-200;
    realcell20=cell+cell20_diff;

    [cell20,cell,realcell20,cell20_diff,crecord,length] = ...
            removeOutliners(2,2,1.1,cell,realcell20,cell20,crecord);
end

%%
%%设置初始std_cell和std_realcell20
std_cell = cell;
std_realcell20 = realcell20;
std_cell20 = cell20;
std_crecord = crecord;
std_cell20_diff = cell20_diff;


%%
%%复原
clear cell cell20 crecord cell20_diff realcell20;
cell=cell_default2;%%反
cell20=cell20_default2;%%反
crecord=crecord_default2;%%反
cell20_diff(:,1) = cell20(:,1)-200;%%小正方形的中心点-大正方形的中心点=diff
cell20_diff(:,2) = cell20(:,2)-200;
realcell20=cell+cell20_diff;

%%
% stdsize = size(std_cell);
% if (stdsize(1,1) < 50)
%     return;%%have a look of current std_data
% %     continue;
% end

%%
%%固定ratio and number
for m = 1:5

    %%with std_plane
    [cell20,cell,realcell20,cell20_diff,crecord,length] = ...
        removeOutliners_std(3,3,1.1,cell,realcell20,cell20,crecord,std_cell,std_realcell20); 
 
    %%without std_plane
    for i = 1:10
        [cell20,cell,realcell20,cell20_diff,crecord,length] = ...
            removeOutliners(1,1,1.1,cell,realcell20,cell20,crecord);
    end
    for i = 1:10
        [cell20,cell,realcell20,cell20_diff,crecord,length] = ...
            removeOutliners(4,4,1.2,cell,realcell20,cell20,crecord);
    end
       
    %%加新的std_cell and std_realcell20
    std_cell = [std_cell;cell];
    std_realcell20 = [std_realcell20;realcell20];
    std_cell20 = [std_cell20;cell20];
    std_crecord = [std_crecord;crecord];
    
    %%通过让std_plane apply function来决定std_plane是否绝对精准
    %%(看std_plane的点有没有明显减少,按理来说是不应该有任何减少的)
    [std_cell, index, ic] = unique(std_cell,'rows','stable');
    std_realcell20 = std_realcell20(index,:);
    std_cell20 = std_cell20(index,:);
    std_crecord = std_crecord(index,:);
    
    %%without std_plane//test std cells
    for i = 1:3
        [std_cell20,std_cell,std_realcell20,std_cell20_diff,std_crecord,length] = ...
            removeOutliners(1,1,1.1,std_cell,std_realcell20,std_cell20,std_crecord);
    end

    %%复原
    clear cell cell20 crecord cell20_diff;
    cell=cell_default2;%%反
    cell20=cell20_default2;%%反
    crecord=crecord_default2;%%反
    cell20_diff(:,1) = cell20(:,1)-200;%%小正方形的中心点-大正方形的中心点=diff
    cell20_diff(:,2) = cell20(:,2)-200;
    realcell20=cell+cell20_diff;
end

%%
%%固定ratio and number
% for m = 1:5
%     
%     %%with std_plane
%     [cell20,cell,realcell20,cell20_diff,crecord,length] = ...
%         removeOutliners_std(1,1,1.08,cell,realcell20,cell20,crecord,std_cell,std_realcell20);
%     
%     %%without std_plane
%     for i = 1:10
%         [cell20,cell,realcell20,cell20_diff,crecord,length] = ...
%             removeOutliners(1,1,1.1,cell,realcell20,cell20,crecord);
%     end
%     for i = 1:10
%         [cell20,cell,realcell20,cell20_diff,crecord,length] = ...
%             removeOutliners(3,3,1.2,cell,realcell20,cell20,crecord);
%     end
%        
%     %%加新的std_cell and std_realcell20
%     std_cell = [std_cell;cell];
%     std_realcell20 = [std_realcell20;realcell20];
%     std_cell20 = [std_cell20;cell20];
%     std_crecord = [std_crecord;crecord];
%     
%     %%通过让std_plane apply function来决定std_plane是否绝对精准
%     %%(看std_plane的点有没有明显减少,按理来说是不应该有任何减少的)
%     [std_cell, index, ic] = unique(std_cell,'rows','stable');
%     std_realcell20 = std_realcell20(index,:);
%     std_cell20 = std_cell20(index,:);
%     std_crecord = std_crecord(index,:);
%     
%     %%without std_plane//test std cells
%     for i = 1:3
%         [std_cell20,std_cell,std_realcell20,std_cell20_diff,std_crecord,length] = ...
%             removeOutliners(1,1,1.1,std_cell,std_realcell20,std_cell20,std_crecord);
%     end
%     
%     %%复原
%     clear cell cell20 crecord cell20_diff;
%     cell=cell_default2;%%反
%     cell20=cell20_default2;%%反
%     crecord=crecord_default2;%%反
%     cell20_diff(:,1) = cell20(:,1)-200;%%小正方形的中心点-大正方形的中心点=diff
%     cell20_diff(:,2) = cell20(:,2)-200;
%     realcell20=cell+cell20_diff;
% end

%%
%%正常 image
imshow(image19rotate)
hold on
plot(cell(:,1),cell(:,2),'yellow*')
hold

figure
imshow(image20)
hold on
plot(realcell20(:,1),realcell20(:,2),'yellow*')
hold off

%%
%%Std image
imshow(image19rotate)
hold on
plot(std_cell(:,1),std_cell(:,2),'yellow*')
hold

figure
imshow(image20)
hold on
plot(std_realcell20(:,1),std_realcell20(:,2),'yellow*')
hold off

%%
%%prepare for std_cell interpolation
cell = std_cell;
realcell20 = std_realcell20;
cell20_diff = realcell20-cell;

%
x = double(cell(:,1));
y = double(cell(:,2));
v1 = double(cell20_diff(:,1));

%
F1 = scatteredInterpolant(x,y,v1);

%
[xq,yq] = meshgrid(1:1:3500);
F1.Method = 'natural';
vq1 = F1(xq,yq);

%
figure
plot3(x,y,v1,'mo')
hold on
mesh(xq,yq,vq1)
title('Natural Neighbor')

%
x = double(cell(:,1));
y = double(cell(:,2));
v2 = double(cell20_diff(:,2));

%
F2 = scatteredInterpolant(x,y,v2);

%
[xq,yq] = meshgrid(1:1:3500);
F2.Method = 'natural';
vq2 = F2(xq,yq);

%
figure
plot3(x,y,v2,'mo')
hold on
mesh(xq,yq,vq2)
title('Natural Neighbor')

%
newx1 = double(xq-round(vq1));%%method2--20-change=19
newy1 = double(yq-round(vq2));

%
newimage191(1:3500,1:3500,1:3) = 0;%%method2--20-change=19
for k = 1:3500
    for j =1:3500
        for s = 1:3
            if (newx1(k,j)>0 & newx1(k,j)<3501 & newy1(k,j)>0 &newy1(k,j)<3501)
            newimage191(k,j,s) = image19rotate(newy1(k,j),newx1(k,j),s);%%要改图片
            end
        end
    end
end
newimage191 = uint8(newimage191)%%是uint8，不是unit8

%%
% imshow(newimage191);
% figure
% imshow(image20)

%%
% imshowpair(newimage191,image20,'checkerboard');

%%
% imshowpair(image19rotate,image20,'checkerboard');

%%
% imshowpair(image19,image20,'checkerboard');

%%
newimage191test = imcrop(newimage191,[251 251 2999 2999]);
image19test = imcrop(image19,[251 251 2999 2999]);
image20test = imcrop(image20,[251 251 2999 2999]);

% imshow(newimage191test);
% figure
% imshow(image19test)

%%
% imshowpair(newimage191test,image20test,'checkerboard');

%%
filename = strcat(targetim,'_', referenceim);
save (filename);

%%
welldonename = strcat(targetim,'well.jpg');
rawname = strcat(targetim,'raw.jpg');
testname = strcat(targetim,'test.jpg');
imwrite(newimage191test,welldonename);
imwrite(image19test,rawname);
imwrite(image20test,testname);

%%
end

%%
function [NMI]=MutualInformation(img1,img2)
        [Ma,Na] = size(img1);
        [Mb,Nb] = size(img2);
        m=min(Ma,Mb);
        n=min(Na,Nb); 
        ET=entropy(img1);
        ES=entropy(img2);%//???
        histq=zeros(256,256);%//????????
        %//?????
        for s=1:m
            for t=1:n
                x=img1(s,t)+1;y=img2(s,t)+1;%//??<—>??
                histq(x,y)=histq(x,y)+1;
            end
        end
        p=histq./sum(sum(histq));%//??????
        EST=-sum(sum(p.*log(p+eps)));
        NMI=(ES+ET)-EST;
end

%%
function [cell20,cell,realcell20,cell20_diff,crecord,length] = removeOutliners(number,rightnumber,ratio,cell,realcell20,cell20,crecord)


%%
length = size(cell);
length = length(1,1);
disp(length);


%%
knnnumber = number+1;
[mIdx,mD] = knnsearch(cell,cell,'K',knnnumber,'Distance','euclidean');


%%
clear rawdiff rawdiffline xdiff xdiff;
rawdiff = [];
rawdiffline = [];
xdiff = [];
ydiff = [];
for k = 1:length
    for j = 2:knnnumber
        xdiff = cell(mIdx(k,1),1)-cell(mIdx(k,j),1);
        ydiff = cell(mIdx(k,1),2)-cell(mIdx(k,j),2);
        rawdiffline = [rawdiffline,xdiff,ydiff];
    end
    rawdiff(k,:) = rawdiffline;
    rawdiffline = [];
end
disp(length);


%%
clear realrawdiff realrawdiffline xdiff ydiff;
realrawdiff = [];
realrawdiffline = [];
xdiff1 = [];
ydiff1 = [];
for k = 1:length
    for j = 2:knnnumber
        xdiff = realcell20(mIdx(k,1),1)-realcell20(mIdx(k,j),1);
        ydiff = realcell20(mIdx(k,1),2)-realcell20(mIdx(k,j),2);
        realrawdiffline = [realrawdiffline,xdiff,ydiff];
    end
    realrawdiff(k,:) = realrawdiffline;
    realrawdiffline = [];
end
disp(length);


%%
clear upxdiff lowxdiff diffline diff;
upxdiff = [];
lowxdiff = [];
diffline = [];
diff = [];
doublenumber = number.*2;
for k = 1:length
    diffline = [];
    for j = 1:doublenumber
        upxdiff = rawdiff(k,j).*ratio;
        lowxdiff = rawdiff(k,j).*(2-ratio);
        diffline = [diffline,upxdiff,lowxdiff];
    end
    diff(k,:) = diffline;
end
disp(length);

%%
for k = 1:length
    for j = 1:doublenumber
        twoj = j.*2;
        if (rawdiff(k,j)<0)
        a = diff(k,twoj-1);
        diff(k,twoj-1) = diff(k,twoj);
        diff(k,twoj) = a;
        end
    end
end
disp(length);

%%
clear index;
doublerightnumber = rightnumber.*2;
total = 0;
index=[];
for k=1:length
    total = 0;
    for j = 1:doublenumber
        twoj = j.*2;
        if (realrawdiff(k,j)<=diff(k,twoj-1) & realrawdiff(k,j)>=diff(k,twoj))
            total = total + 1;
        end
    end
    if (total >= doublerightnumber)
        index = [index,k]
    end
end
index = unique(index);
cell20 = cell20(index,:);
cell = cell(index,:);
crecord = crecord(index,:);
realcell20 = realcell20(index,:);
disp(length);

length = size(cell);
length = length(1,1);
disp(length);


%%
clear cell20_diff;
cell20_diff(:,1) = cell20(:,1)-200;%%小正方形的中心点-大正方形的中心点=diff
cell20_diff(:,2) = cell20(:,2)-200;



end

%%
function [cell20,cell,realcell20,cell20_diff,crecord,length] = removeOutliners_std(number,rightnumber,ratio,cell,realcell20,cell20,crecord,std_cell,std_realcell20)


%%
length = size(cell);
length = length(1,1);


%%
[mIdx,mD] = knnsearch(std_cell,cell,'K',number,'Distance','euclidean');


%%
clear rawdiff rawdiffline xdiff xdiff;
rawdiff = [];
rawdiffline = [];
xdiff = [];
ydiff = [];
for k = 1:length
    for j = 1:number
        xdiff = cell(k,1)-std_cell(mIdx(k,j),1);
        ydiff = cell(k,2)-std_cell(mIdx(k,j),2);
        rawdiffline = [rawdiffline,xdiff,ydiff];
    end
    rawdiff(k,:) = rawdiffline;
    rawdiffline = [];
end


%%
clear realrawdiff realrawdiffline xdiff ydiff;
realrawdiff = [];
realrawdiffline = [];
xdiff1 = [];
ydiff1 = [];
for k = 1:length
    for j = 1:number
        xdiff = realcell20(k,1)-std_realcell20(mIdx(k,j),1);
        ydiff = realcell20(k,2)-std_realcell20(mIdx(k,j),2);
        realrawdiffline = [realrawdiffline,xdiff,ydiff];
    end
    realrawdiff(k,:) = realrawdiffline;
    realrawdiffline = [];
end


%%
clear upxdiff lowxdiff diffline diff;
upxdiff = [];
lowxdiff = [];
diffline = [];
diff = [];
doublenumber = number.*2;
for k = 1:length
    diffline = [];
    for j = 1:doublenumber
        upxdiff = rawdiff(k,j).*ratio;
        lowxdiff = rawdiff(k,j).*(2-ratio);
        diffline = [diffline,upxdiff,lowxdiff];
    end
    diff(k,:) = diffline;
end


%%
for k = 1:length
    for j = 1:doublenumber
        twoj = j.*2;
        if (rawdiff(k,j)<0)
        a = diff(k,twoj-1);
        diff(k,twoj-1) = diff(k,twoj);
        diff(k,twoj) = a;
        end
    end
end


%%
clear index;
doublerightnumber = rightnumber.*2;
total = 0;
index=[];
for k=1:length
    total = 0;
    for j = 1:doublenumber
        twoj = j.*2;
        if (realrawdiff(k,j)<=diff(k,twoj-1) & realrawdiff(k,j)>=diff(k,twoj))
            total = total + 1;
        end
    end
    if (total >= doublerightnumber)
        index = [index,k]
    end
end
index = unique(index);
cell20 = cell20(index,:);
cell = cell(index,:);
crecord = crecord(index,:);

length = size(cell);
length = length(1,1);


%%
clear cell20_diff realcell20;
cell20_diff(:,1) = cell20(:,1)-200;%%小正方形的中心点-大正方形的中心点=diff
cell20_diff(:,2) = cell20(:,2)-200;
realcell20=cell+cell20_diff;



end

%%
function [] = extract_centroid(img_file, magnification, save_name)

	% Input:
	% img_file			- string, image file to be processed.
	% magnification		- integer.
	% save_name			- string, file to store centroid mask image.

	img = imread(img_file);
	mask = segNuclei(img, magnification);
	imwrite(mask, save_name);

end

%%
function objCtrMask=segNuclei(img,magnification)

% input: the image of interested region (img)
% output: the centroid point of cell
% Faliu.Yi@UTSouthwestern.edu

%========================0: read image=====================================
%==========================================================================
sampleRGB=img;
[height width channel]=size(sampleRGB);

%========================1: color deconvolution============================
%==========================================================================
SourceImage=sampleRGB;
H_deinterlace = [0 1 0; 0 2 0; 0 1 0] ./4;

sample_deinterlace = zeros(height, width, channel);
for k=1:channel
    sample_deinterlace(:,:,k) = filter2(H_deinterlace,double(sampleRGB(:,:,k)),'same');
end

%=== Convert RGB intensity to optical density (absorbance)
sampleRGB_OD = -log((sample_deinterlace+1)./256);

%% Construct color deconvolution matrix
H2 = ones(10,10) ./ 100;
sampleRGB_OD_Blur = zeros(height,width,channel);

for k=1:channel
    sampleRGB_OD_Blur(:,:,k) = filter2(H2,sampleRGB_OD(:,:,k),'same');
end

% Standard values from literature
He = [0.550 0.758 0.351]';
Eo = [0.398 0.634 0.600]';
Bg = [0.754 0.077 0.652]';

% Create Deconvolution matrix
M = [He/norm(He) Eo/norm(Eo) Bg/norm(Bg)];
D = inv(M);

% Apply Color Deconvolution
sampleHEB_OD = zeros(height, width, channel);
for i=1:height
    for j=1:width
        RGB = reshape(sampleRGB_OD(i,j,:),channel,1);
        HEB = D * RGB;

      	sampleHEB_OD(i,j,1) = HEB(1);
       	sampleHEB_OD(i,j,2) = HEB(2);
       	sampleHEB_OD(i,j,3) = HEB(3);
    end
end

% Extract tumor cells that are stained by hematoxylin only
hematoxylin = sampleHEB_OD(:,:,1);
hematoxylin = (hematoxylin - 0.05) .* (hematoxylin > 0.05);
hematoxylin = hematoxylin ./ max(max(hematoxylin));
h=fspecial('sobel');
hematoxylin_grad=sqrt(imfilter(hematoxylin,h,'replicate').^2+imfilter(hematoxylin,h','replicate').^2);


%=================2: morphological operstions==============================
%==========================================================================
hImg=hematoxylin;

se1=strel('disk',4);              %opening by reconstruction: remove small object
hImgEro=imerode(hImg,se1);
hImgEroRec=imreconstruct(hImgEro,hImg);

se2=strel('disk',7);
hImgDia=imdilate(hImgEroRec,se2);  %closing by reconstruction: remove small black spot
hImgDiaRec=imreconstruct(imcomplement(hImgDia),imcomplement(hImgEroRec));
hImgDiaRec=imcomplement(hImgDiaRec);
hImgFill=imfill(hImgDiaRec,'holes');


%=======================3:Thresholding=====================================
%==========================================================================
T1=graythresh(hImgFill);
hImgThres=(hImgFill>T1);
hImgThresFill=imfill(hImgThres,'holes');
exMaskAdd=imcomplement(hImgThresFill);  % used as additional external markers
se22=strel('disk',3);
exMaskAdd=imerode(exMaskAdd,se22);
se3=strel('disk',2);
hImgClose=imopen(hImgThresFill,se3);   

hImgAreaOpen=bwareaopen(hImgClose,30); 



distInfo=bwdist(imcomplement(hImgAreaOpen));
inMark=distInfo>2;
inMark=bwareaopen(inMark,5);          
dist2=bwdist(inMark);
inSeg=watershed(dist2);
exMark=inSeg==0;                      


se4=strel('disk',1);                  
inMark=imerode(inMark,se4);
markers=inMark | exMark | exMaskAdd;              

%===========================4: Level Set==================================
%=========================================================================

Img = double(hImgFill);
[row,col] = size(Img);

phi = ones(row,col);
phi(exMark|exMaskAdd)=-1;
u=phi;

sigma = 1;
G = fspecial('gaussian', 11, sigma);   %5

delt = 1;
Iter = 30;     
mu=60;

for n = 1:Iter
    [ux, uy] = gradient(u);
   
    c1 = sum(sum(Img.*(u<0)))/(sum(sum(u<0)));
    c2 = sum(sum(Img.*(u>=0)))/(sum(sum(u>=0)));
    
    spf = Img - (c1 + c2)/2;
    spf = spf/(max(abs(spf(:))));
    
    u = u + delt*(mu*spf.*sqrt(ux.^2 + uy.^2));
    u = (u >= 0) - ( u< 0);
    u = conv2(u, G, 'same');
end

temp=im2bw(u,0.8);
temp=bwareaopen(temp,26);  %remove small region



imgMask=temp;   % from level set
imgMaskDist=bwdist(imcomplement(imgMask));
imgDistMax=imregionalmax(imgMaskDist);

if magnification==40
se5=strel('disk',8);
imgDistMax=imdilate(imgDistMax,se5);

elseif magnification==20
se5=strel('disk',4);
imgDistMax=imdilate(imgDistMax,se5);
end



[label1,num1]=bwlabel(imgDistMax);
prop1=regionprops(label1,'Centroid');
ctrPos1=cat(1,prop1.Centroid);

xp=ctrPos1(:,1);
yp=ctrPos1(:,2);

ctrRow=round(yp);
ctrCol=round(xp);

objCtrMaskTemp=zeros(height,width);
indexTemp=(ctrCol-1)*height+ctrRow;
objCtrMaskTemp(indexTemp)=1;

objCtrMask=objCtrMaskTemp;




%=========================separate overlapping region ==============================
if magnification==40
   S1=fastradial(hImgFill,7,2,0.01);
   S1=imcomplement(S1);

   regMin1=imhmin(S1,0.01);
   regMin1=imregionalmin(regMin1);
   se=strel('disk',1);
   regMin1=imdilate(regMin1,se);
%    figure
%    imshow(regMin1)
%    title('radial symmetry with size 8')

   S2=fastradial(hImgFill,9,2,0.01);
   S2=imcomplement(S2);

   regMin2=imhmin(S2,0.01);
   regMin2=imregionalmin(regMin2);
   se=strel('disk',1);
   regMin2=imdilate(regMin2,se);
%    figure
%    imshow(regMin2)
%    title('radial symmetry with size 11')

   S3=fastradial(hImgFill,13,2,0.01);
   S3=imcomplement(S3);

   regMin3=imhmin(S3,0.01);
   regMin3=imregionalmin(regMin3);
   se=strel('disk',1);
   regMin3=imdilate(regMin3,se);
%    figure
%    imshow(regMin3)
%    title('radial symmetry with size 15')

   S4=fastradial(hImgFill,17,2,0.01);
   S4=imcomplement(S4);

   regMin4=imhmin(S4,0.01);
   regMin4=imregionalmin(regMin4);
   se=strel('disk',1);
   regMin4=imdilate(regMin4,se);
%    figure
%    imshow(regMin4)
%    title('radial symmetry with size 19')

   S5=fastradial(hImgFill,21,2,0.01);
   S5=imcomplement(S5);

   regMin5=imhmin(S5,0.01);
   regMin5=imregionalmin(regMin5);
   se=strel('disk',1);
   regMin5=imdilate(regMin5,se);
%    figure
%    imshow(regMin5)
%    title('radial symmetry with size 25')

   for iRF=1:4
      eval(['rfMarkerTemp=regMin' num2str(iRF) '& regMin' num2str(iRF+1) ';']);
      eval(['im' num2str(iRF) '=imreconstruct(rfMarkerTemp,regMin' num2str(iRF) ');']);
      eval(['im' num2str(iRF+1) '=imreconstruct(rfMarkerTemp,regMin' num2str(iRF+1) ');']);
      eval(['regMin' num2str(iRF+1) '=rfMarkerTemp+(regMin' num2str(iRF) '-im' num2str(iRF) ')+(regMin' num2str(iRF+1) '-im' num2str(iRF+1) ');'])
      eval(['regMin' num2str(iRF+1) '=logical(regMin' num2str(iRF+1) ');'])
   end
   rFinMarker=regMin5;
%    imgPerm=bwperim(rFinMarker);
%    overlay2=imoverlay(sampleRGB,imgPerm,[.3 1 .3]);
%    figure
%    imshow(overlay2)
%    title('internal markers from RF')

elseif magnification==20
   S1=fastradial(hImgFill,4,2,0.01);
   S1=imcomplement(S1);

   regMin1=imhmin(S1,0.01);
   regMin1=imregionalmin(regMin1);
   se=strel('disk',1);
   regMin1=imdilate(regMin1,se);
%    figure
%    imshow(regMin1)
%    title('radial symmetry with size 4')

   S2=fastradial(hImgFill,6,2,0.01);
   S2=imcomplement(S2);

   regMin2=imhmin(S2,0.01);
   regMin2=imregionalmin(regMin2);
   se=strel('disk',1);
   regMin2=imdilate(regMin2,se);
%    figure
%    imshow(regMin2)
%    title('radial symmetry with size 6')

   S3=fastradial(hImgFill,8,2,0.01);
   S3=imcomplement(S3);

   regMin3=imhmin(S3,0.01);
   regMin3=imregionalmin(regMin3);
   se=strel('disk',1);
   regMin3=imdilate(regMin3,se);
%    figure
%    imshow(regMin3)
%    title('radial symmetry with size 8')

   S4=fastradial(hImgFill,10,2,0.01);
   S4=imcomplement(S4);

   regMin4=imhmin(S4,0.01);
   regMin4=imregionalmin(regMin4);
   se=strel('disk',1);
   regMin4=imdilate(regMin4,se);
%    figure
%    imshow(regMin4)
%    title('radial symmetry with size 10')

   S5=fastradial(hImgFill,11,2,0.01);
   S5=imcomplement(S5);

   regMin5=imhmin(S5,0.01);
   regMin5=imregionalmin(regMin5);
   se=strel('disk',1);
   regMin5=imdilate(regMin5,se);
%    figure
%    imshow(regMin5)
%    title('radial symmetry with size 11')

   for iRF=1:4
      eval(['rfMarkerTemp=regMin' num2str(iRF) '& regMin' num2str(iRF+1) ';']);
      eval(['im' num2str(iRF) '=imreconstruct(rfMarkerTemp,regMin' num2str(iRF) ');']);
      eval(['im' num2str(iRF+1) '=imreconstruct(rfMarkerTemp,regMin' num2str(iRF+1) ');']);
      eval(['regMin' num2str(iRF+1) '=rfMarkerTemp+(regMin' num2str(iRF) '-im' num2str(iRF) ')+(regMin' num2str(iRF+1) '-im' num2str(iRF+1) ');'])
      eval(['regMin' num2str(iRF+1) '=logical(regMin' num2str(iRF+1) ');'])
   end
   rFinMarker=regMin5;
end

if magnification==40
   se1=strel('disk',5);
   imgMaskRF=imdilate(rFinMarker,se1);
elseif magnification==20
    se1=strel('disk',3);
    imgMaskRF=imdilate(rFinMarker,se1);
end

[label1 num1]=bwlabel(imgMaskRF);
prop1=regionprops(label1,'Centroid');
ctrPos1=cat(1,prop1.Centroid);

xp=ctrPos1(:,1);
yp=ctrPos1(:,2);

ctrRow=round(yp);
ctrCol=round(xp);

objCtrMaskTemp=zeros(height,width);
indexTemp=(ctrCol-1)*height+ctrRow;
objCtrMaskTemp(indexTemp)=1;

objCtrMask2=objCtrMaskTemp;              % dot representation from RF

[ypf xpf]=find(objCtrMask2==1);


[label1,num1]=bwlabel(imgMask);
cellProp1=regionprops(label1,'area');
cellArea=[cellProp1.Area];
[aValue aIndex]=sort(cellArea,'descend');

bigAreaMask=logical(zeros(height,width));

if length(aIndex)~=0
   if magnification==40
       bigIndex=aValue>10000;
   elseif magnification==20
       bigIndex=aValue>2500;
   end
   
   bigNum=sum(bigIndex);
   
   if bigNum~=0
%       bigAreaMask=logical(zeros(height,width));
      for ibigNum=1:bigNum
          bigAreaIndex=aIndex(ibigNum);
          bigAreaMaskTemp=label1==bigAreaIndex;
          bigAreaMask=bigAreaMask | bigAreaMaskTemp;
      end
   end   
end

objCtrMask3=objCtrMask2 & bigAreaMask;


bigAreaMaskComp=imcomplement(bigAreaMask);
objCtrMask4=bigAreaMaskComp & objCtrMask;
objCtrMask=objCtrMask3 | objCtrMask4;
if magnification==40
    se=strel('disk',5);
    objCtrMask=imdilate(objCtrMask,se);
elseif magnification==20
    se=strel('disk',3);
    objCtrMask=imdilate(objCtrMask,se);
end

if sum(sum(objCtrMask))~=0

[label1,num1]=bwlabel(objCtrMask);
prop1=regionprops(label1,'Centroid');
ctrPos1=cat(1,prop1.Centroid);

xp=ctrPos1(:,1);
yp=ctrPos1(:,2);

ctrRow=round(yp);
ctrCol=round(xp);

objCtrMaskTemp=zeros(height,width);
indexTemp=(ctrCol-1)*height+ctrRow;
objCtrMaskTemp(indexTemp)=1;

objCtrMask=objCtrMaskTemp;  
end

% figure
% imshow(objCtrMask)
% title('final dot representation')

% [yp xp]=find(objCtrMask==1);
% figure
% colormap(gray(256)), imagesc(sampleRGB);
% axis off
% hold on
% plot(xp,yp,'.r')
% hold off

end

%%
% FASTRADIAL - Loy and Zelinski's fast radial feature detector
%
% An implementation of Loy and Zelinski's fast radial feature detector
%
% Usage: S = fastradial(im, radii, alpha, beta)
%
% Arguments:
%            im    - Image to be analysed
%            radii - Array of integer radius values to be processed
%                    suggested radii might be [1 3 5]
%            alpha - Radial strictness parameter.
%                    1 - slack, accepts features with bilateral symmetry.
%                    2 - a reasonable compromise.
%                    3 - strict, only accepts radial symmetry.
%                        ... and you can go higher
%            beta  - Gradient threshold.  Gradients below this threshold do
%                    not contribute to symmetry measure, defaults to 0.
%
% Returns    S     - Symmetry map.  Bright points with high symmetry are
%                    marked with large positive values. Dark points of
%                    high symmetry marked with large -ve values.
%
% To localize points use NONMAXSUPPTS on S, -S or abs(S) depending on
% what you are seeking to find.

% Reference:
% Loy, G.  Zelinsky, A.  Fast radial symmetry for detecting points of
% interest.  IEEE PAMI, Vol. 25, No. 8, August 2003. pp 959-973.

% Copyright (c) 2004-2010 Peter Kovesi
% Centre for Exploration Targeting
% The University of Western Australia
% http://www.csse.uwa.edu.au/~pk/research/matlabfns/
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in 
% all copies or substantial portions of the Software.
%
% The Software is provided "as is", without warranty of any kind.

% November 2004  - original version
% July     2005  - Bug corrected: magitude and orientation matrices were
%                  not zeroed for each radius value used (Thanks to Ben
%                  Jackson) 
% December 2009  - Gradient threshold added + minor code cleanup
% July     2010  - Gradients computed via Farid and Simoncelli's 5 tap
%                  derivative filters

function [S, So] = fastradial(im, radii, alpha, beta, feedback)
    
    if ~exist('beta','var'),     beta = 0;     end
    if ~exist('feedback','var'), feedback = 0; end    
    
    if any(radii ~= round(radii)) || any(radii < 1)
        error('radii must be integers and > 1')
    end
    
    [rows,cols]=size(im);
    
    % Compute derivatives in x and y via Farid and Simoncelli's 5 tap
    % derivative filters
    [imgx, imgy] = derivative5(im, 'x', 'y');
    mag = sqrt(imgx.^2 + imgy.^2)+eps; % (+eps to avoid division by 0)
    
    % Normalise gradient values so that [imgx imgy] form unit 
    % direction vectors.
    imgx = imgx./mag;   
    imgy = imgy./mag;
    
    S = zeros(rows,cols,numel(radii));  % Symmetry matrix
    So = zeros(rows,cols,numel(radii)); % Orientation only symmetry matrix    
    
    [x,y] = meshgrid(1:cols, 1:rows);
    
    for n = radii
        M = zeros(rows,cols);  % Magnitude projection image
        O = zeros(rows,cols);  % Orientation projection image

        % Coordinates of 'positively' and 'negatively' affected pixels
        posx = x + round(n*imgx);
        posy = y + round(n*imgy);
        
        negx = x - round(n*imgx);
        negy = y - round(n*imgy);
        
        % Clamp coordinate values to range [1 rows 1 cols]
        posx( posx<1 )    = 1;
        posx( posx>cols ) = cols;
        posy( posy<1 )    = 1;
        posy( posy>rows ) = rows;
        
        negx( negx<1 )    = 1;
        negx( negx>cols ) = cols;
        negy( negy<1 )    = 1;
        negy( negy>rows ) = rows;
        
        I = sub2ind( [rows,cols], posy, posx);
        O(:) = accumarray( I(:), ones(size(I(:))), [prod([rows,cols]),1], @sum);
        M(:) = accumarray( I(:), mag(:), [prod([rows,cols]),1], @sum);
        I(:) = sub2ind( [rows,cols], negy, negx);
        O(:) = O(:) - accumarray( I(:), ones(size(I(:))), [prod([rows,cols]),1], @sum);
        M(:) = M(:) - accumarray( I(:), mag(:), [prod([rows,cols]),1], @sum);
        
        % Clamp Orientation projection matrix values to a maximum of 
        % +/-kappa,  but first set the normalization parameter kappa to the
        % values suggested by Loy and Zelinski
        if n == 1, kappa = 8; else kappa = 9.9; end
        
        O(O >  kappa) =  kappa;  
        O(O < -kappa) = -kappa;  
        
        % Unsmoothed symmetry measure at this radius value
        F = M./kappa .* (abs(O)/kappa).^alpha;
        Fo = sign(O) .* (abs(O)/kappa).^alpha;   % Orientation only based measure
        
        % Smooth and spread the symmetry measure with a Gaussian proportional to
        % n.  Also scale the smoothed result by n so that large scales do not
        % lose their relative weighting.
        S(:,:,n==radii) = gaussfilt(F,  0.25*n) * n;
        So(:,:,n==radii) = gaussfilt(Fo, 0.25*n) * n;        
    end  % for each radius
    
%     S  = S /length(radii);  % Average
%     So = So/length(radii); 
end

%%
function varargout = derivative5(im, varargin) 

 
     varargin = varargin(:); 
     varargout = cell(size(varargin)); 
      
     % Check if we are just computing 1st derivatives.  If so use the 
     % interpolant and derivative filters optimized for 1st derivatives, else 
     % use 2nd derivative filters and interpolant coefficients. 
     % Detection is done by seeing if any of the derivative specifier 
     % arguments is longer than 1 char, this implies 2nd derivative needed. 
     secondDeriv = false;     
     for n = 1:length(varargin) 
         if length(varargin{n}) > 1 
             secondDeriv = true; 
             break 
         end 
     end 
      
     if ~secondDeriv 
         % 5 tap 1st derivative cofficients.  These are optimal if you are just 
         % seeking the 1st deriavtives 
         p = [0.037659  0.249153  0.426375  0.249153  0.037659]; 
         d1 =[0.109604  0.276691  0.000000 -0.276691 -0.109604]; 
     else          
         % 5-tap 2nd derivative coefficients. The associated 1st derivative 
         % coefficients are not quite as optimal as the ones above but are 
         % consistent with the 2nd derivative interpolator p and thus are 
         % appropriate to use if you are after both 1st and 2nd derivatives. 
         p  = [0.030320  0.249724  0.439911  0.249724  0.030320]; 
         d1 = [0.104550  0.292315  0.000000 -0.292315 -0.104550]; 
         d2 = [0.232905  0.002668 -0.471147  0.002668  0.232905]; 
     end 
 
 
     % Compute derivatives.  Note that in the 1st call below MATLAB's conv2 
     % function performs a 1D convolution down the columns using p then a 1D 
     % convolution along the rows using d1. etc etc. 
     gx = false; 
      
     for n = 1:length(varargin) 
       if strcmpi('x', varargin{n}) 
           varargout{n} = conv2(p, d1, im, 'same');     
           gx = true;   % Record that gx is available for gxy if needed 
           gxn = n; 
       elseif strcmpi('y', varargin{n}) 
           varargout{n} = conv2(d1, p, im, 'same'); 
       elseif strcmpi('xx', varargin{n}) 
           varargout{n} = conv2(p, d2, im, 'same');     
       elseif strcmpi('yy', varargin{n}) 
           varargout{n} = conv2(d2, p, im, 'same'); 
       elseif strcmpi('xy', varargin{n}) | strcmpi('yx', varargin{n}) 
           if gx 
              varargout{n} = conv2(d1, p, varargout{gxn}, 'same'); 
           else 
               gx = conv2(p, d1, im, 'same');     
               varargout{n} = conv2(d1, p, gx, 'same'); 
           end 
       else 
           error(sprintf('''%s'' is an unrecognized derivative option',varargin{n})); 
       end 
     end 
end

%%
% GAUSSFILT -  Small wrapper function for convenient Gaussian filtering
%
% Usage:  smim = gaussfilt(im, sigma)
%
% Arguments:   im - Image to be smoothed.
%           sigma - Standard deviation of Gaussian filter.
%
% Returns:   smim - Smoothed image.
%
% See also:  INTEGGAUSSFILT

% Peter Kovesi
% Centre for Explortion Targeting
% The University of Western Australia
% http://www.csse.uwa.edu.au/~pk/research/matlabfns/

% March 2010

function smim = gaussfilt(im, sigma)
 
    assert(ndims(im) == 2, 'Image must be greyscale');
    
    % If needed convert im to double
    if ~strcmp(class(im),'double')
        im = double(im);  
    end
    
    sze = ceil(6*sigma);  
    if ~mod(sze,2)    % Ensure filter size is odd
        sze = sze+1;
    end
    sze = max(sze,1); % and make sure it is at least 1
    
    h = fspecial('gaussian', [sze sze], sigma);

    smim = filter2(h, im);

end