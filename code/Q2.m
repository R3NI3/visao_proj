function [result] = main()
	img = imread('~/Workspace/visao/imagens/35008.jpg');
	g_img = rgb2gray(img);
	h1 = fspecial('gaussian', 5, 0.5);
	h2 = fspecial('gaussian', 5, 1.5);%1.5
	h = h2 - h1;
	%difference of gaussian
	result = imfilter(g_img, h);
	result = im2bw(result,0.01);
	%imshow(result);

	%dilatation followed by erosion to eliminate noise inside img(not perfect)
	SE1 = strel('square',13);%11
	SE2 = strel('square',11);
	result = imdilate(result, SE2);
	result = imerode(result, SE1);
	%figure;
	%imshow(result);

	%Apply mask found to rgb image
	R = img(:,:,1); % REd
	G = img(:,:,2); % Green
	B = img(:,:,3); % Blue
	new_img = cat(3,uint8(result).*R, uint8(result).*G, uint8(result).*B);
	%figure;
	imshow(new_img);
end