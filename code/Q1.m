function [result] = main()
	tmp = imread('~/Workspace/visao/imagens/parafuso_porca.bmp');
	test_img = imread('~/Workspace/visao/imagens/objetos.bmp');
	%limiarize and complement image
	tmp = imcomplement(im2bw(tmp,0.7));
	test_img = imcomplement(im2bw(test_img,0.7));
	%find the proportions of the searched objects
	objs_rate = obj_rate_search(tmp);
	 %look for objects in test image that have similar desired proportions
	result = obj_search(test_img, objs_rate);

	%object proportions , num of objects found
	result = {'Porca: ', num2str(result(1));
				'Parafuso: ', num2str(result(2))};
end

function [objs_rate] = obj_rate_search(img)
	objs_rate = [];
	objs = regionprops(img,'MajorAxisLength','MinorAxisLength');

	for obj = objs'
		maj = obj.MajorAxisLength;
		minor = obj.MinorAxisLength;
		rate = max((maj/minor),(minor/maj));
		objs_rate = [objs_rate, rate];
	end
end

function [num_objs] = obj_search(img, objs_rate)
	[m,n] = size(objs_rate);
	num_objs = zeros(1,n);
	objs = regionprops(img,'MajorAxisLength','MinorAxisLength');

	for obj = objs'
		maj = obj.MajorAxisLength;
		minor = obj.MinorAxisLength;
		rate = max((maj/minor),(minor/maj));
		for ra_idx = [1:1:n]
			if (rate <= 1.1*objs_rate(ra_idx) && rate >= 0.9*objs_rate(ra_idx))
				num_objs(ra_idx) = num_objs(ra_idx)+1;
			end
		end
	end
end