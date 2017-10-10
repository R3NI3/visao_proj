function main()
	%img = imread('~/Workspace/visao/imagens/113044.jpg');
	img = imread('~/Workspace/visao/imagens/predio.bmp');
	method3(img);
end

function method3(img)
	[h,s,v] = rgb2hsv(img);

	[row,col] = find(s < 0.2);

	v(:) = 0;
	for idx = [1:1:size(row)]
		v(row(idx),col(idx)) = 1;
	end

	M = mean(h(:));
	[row,col] = find(h > 1.2*M);
	for idx = [1:1:size(row)]
		v(row(idx),col(idx)) = 0;
	end

	hsv_img = cat(3, h,s,v);
	imshow([h,s,v]);
	n_img = hsv2rgb(hsv_img);
	bw_img = im2bw(n_img);

	SE = strel('square',3);
	bw_img = imerode(bw_img,SE);
	bw_img = imdilate(bw_img,SE);

	figure;
	imshow(bw_img);

	figure;
	%img = imcomplement(bw_img);
	img = edge(bw_img,'Sobel');
	[H,theta,rho] = hough(img);
	peaks  = houghpeaks(H,50);
	lines = houghlines(img,theta,rho,peaks);
	imshow(img)
	hold on
	for k = 1:numel(lines)
		x1 = lines(k).point1(1);
		y1 = lines(k).point1(2);
		x2 = lines(k).point2(1);
		y2 = lines(k).point2(2);
		plot([x1 x2],[y1 y2],'Color','g','LineWidth', 1)
	end
end
