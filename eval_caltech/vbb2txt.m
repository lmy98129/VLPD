setlen = [15,6,12,13,12,13,19,12,11,12,12];

s = [];
set = [];
for i = 1 : 11
   set = ['set',num2str(i - 1,'%02d')] 
   for j = 1 : setlen(i)
       rddir = [];
       sadir = [];
       s = num2str(j - 1,'%02d');
       rdfilename = ['V0',s,'.vbb'];
       safilename = ['V0',s,'.txt'];
       rddir = [set,'\',rdfilename];
       sadir = [set,'\',safilename];
       vbb2txt_func(rddir,sadir);
	end
end

function vbb2txt_func(vbbName,txtName)  
 
vPath = '';
finalPath = [vPath,vbbName]
outpath = '';


%disp(finalPath);
A = vbb('vbbLoad', finalPath);
c = fopen([outpath,'\',txtName],'wt');

for i = 1:A.nFrame    
    iframe = A.objLists(1,i);    
    iframe_data = iframe{1,1};    
    n1length = length(iframe_data);    
    for  j = 1:n1length    
        iframe_dataj = iframe_data(j);    
        if iframe_dataj.pos(1) ~= 0  %pos  posv  
            fprintf(c,'%d %f %f %f %f\n', i, iframe_dataj.pos(1),...  
            iframe_dataj.pos(2),iframe_dataj.pos(3),iframe_dataj.pos(4));    
        end    
    end    
end

end

