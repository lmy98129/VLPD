[r, c] = size(gt);
fp=fopen('gt.txt','wt');
for j =1:c
    current = gt(1, j);
    [w, h] = size(current);
    if w > 0
        for x = 1:w
            [ lines, objs ] = size(current{x});
            for l = 1:lines
                % fprintf(fp, '%f ', char(current{x}(l, :)));
                fprintf(fp,'%f %f %f %f %f ',current{x}(l, 1), current{x}(l, 2), current{x}(l, 3), current{x}(l, 4), current{x}(l, 5));
            end;
        end;
        fprintf(fp,'\n ');
    end;
end;
fclose(fp);