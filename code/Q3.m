function main()
	%img = imread('~/Workspace/visao/imagens/113044.jpg');
	img = imread('~/Workspace/visao/imagens/69020.jpg');
	%imshow(img)
	method(img);
	test();
end

function [img] = method(img)
	[h,s,v] = rgb2hsv(img);
	[M,F] = mode(h(:));
	[row,col] = find(h <= M*1.5 & h >= M*0.7);

	for idx = [1:1:size(row)]
		v(row(idx),col(idx)) = 0;
	end
	%s = im2bw(s,1);
	%s = imcomplement(s);

	hsv_img = cat(3, h,s,v);


	n_img = hsv2rgb(hsv_img);
	g_img = rgb2gray(n_img);
	%imshow(g_img);

	SE = strel('disk',7);
	SE2 = strel('disk',7);
	g_img = imerode(g_img, SE);
	g_img = imdilate(g_img, SE2);

	%figure;
	%imshow(g_img)

	%features = extractLBPFeatures(g_img,'Upright',false);
	%figure;
	%imshow(features);

	CC = bwconncomp(g_img);
	numPixels = cellfun(@numel,CC.PixelIdxList);
	[biggest,idx] = max(numPixels);
	g_img(:) = 0;
	g_img(CC.PixelIdxList{idx}) = 255;
	g_img = im2bw(g_img,0.5);

	imshow(g_img);
	figure
	R = img(:,:,1); % REd
	G = img(:,:,2); % Green
	B = img(:,:,3); % Blue
	new_img = cat(3,uint8(g_img).*R, uint8(g_img).*G, uint8(g_img).*B);
	imshow(new_img);

end

function test()
	img1 = imread('~/Workspace/visao/imagens/113044.jpg');
	img2 = imread('~/Workspace/visao/imagens/42049.jpg');
	figure;
	method(img1);
	figure;
	method(img2);
end