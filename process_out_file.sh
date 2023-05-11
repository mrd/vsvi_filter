#!/bin/bash

outfile="$1"

[ -n "$outfile" ] || exit 1
sqldir="$2"

outshfile="$outfile.sh"
npzfile="`grep 'Loading ' "$outfile" | sed -e 's/.*"\(.*\)"./\1/'`"
jpgfile="`echo $npzfile | rev | cut -f 2- -d. | rev`.jpg"
imgsize="`identify "$jpgfile" | sed -e 's/.*JPEG \([0-9x]*\).*/\1/'`"
imgw=`echo $imgsize | cut -f 1 -d x`
imgh=`echo $imgsize | cut -f 2 -d x`
xwrapneeded=$(($imgw * 7 / 8))
w8=$(($imgw / 8))
w4=$(($imgw / 4))
w98=$(($imgw + $w8))
maxx=$(($imgw + $w4))
h4=$(($imgh / 4))
hFor43Ratio=$(($w4 * 3 / 4))

stem="`echo $jpgfile | cut -f 1 -d.`"
imgid="`echo $stem | rev | cut -f 1 -d/ |rev`"
if [ -n "$sqldir" ]; then
  mkdir -p "$sqldir/`echo $stem | rev | cut -f 2 -d/ |rev`"
  sqlfile="$sqldir/`echo $stem | cut -f 2- -d/`"_update.sql
else
  sqlfile=""
fi

matrixshape=`grep 'Matrix shape' "$outfile" | sed -e 's/.*(\([0-9, ]*\))./\1/'`
matw=`echo $matrixshape | cut -f 2 -d ' '`
xs=`grep 'Found road centres' "$outfile" | sed -e 's/.*\[\([0-9 ]*\)\]./\1/'`

ispano=

if [ ! -n "`grep 'panoramic input' "$outfile"`" ]; then
  echo Stem: $stem, Image ID: $imgid
  xboundlo=$(($matw / 4))
  xboundhi=$(($matw * 3 / 4))
  echo Matrix width: $matw, boundaries: $xboundlo $xboundhi
  echo -n 'Non-panoramic. '
  match=0
  if [ -z "$xs" ]; then
    echo No road found.
  else
    echo Road centres: \[ $xs \].
    for x in $xs; do
      if [[ $x -gt $xboundlo && $x -lt $xboundhi ]]; then
        match=$x
      fi
    done
  fi
  if [ $match != 0 ]; then
    echo Acceptable road centre: $match
    echo "UPDATE image SET usable=true WHERE position('$imgid.jpg' in system_path) > 0;" | tee $sqlfile
  else
    echo "UPDATE image SET usable=false WHERE position('$imgid.jpg' in system_path) > 0;" | tee $sqlfile
  fi
  exit 0
fi

cat > "$outshfile" << 'EOF'
#!/bin/bash
EOF

echo -n | tee $sqlfile

for x in $xs; do
  imgx=$(($imgw * $x / $matw))

  match=0
  wrapx=$(($x - $matw))
  # ugly but effective test for duplicates due to wrap
  for y in $xs; do
    [ $wrapx == $y ] && match=1
  done

  if [ $match == 1 ]; then
    # skip due to duplication
    echo dup
  elif [ $imgx -ge $maxx ]; then
    echo $outfile: $x too large
  elif [ $imgx -ge $w98 ]; then
    # wrapped all the way around
    wrapimgx=$(($imgw * $wrapx / $matw))
    xlo=$(($wrapimgx - $w8))
    crop="-crop ${w4}x${hFor43Ratio}+$xlo+$h4"
    echo convert "$jpgfile" "$crop" "${stem}_x$imgx.jpg" >> "$outshfile"
    imgid2="${imgid}_x$imgx"
    echo "UPDATE image SET usable=true WHERE position('$imgid2.jpg' in system_path) > 0;" | tee -a $sqlfile
  elif [ $imgx -ge $xwrapneeded ]; then
    # must paste together the two sides of the image
    xlo=$(($imgx - $w8))
    w4_p1=$(($imgw - $xlo))
    w4_p2=$(($w4 - $w4_p1))
    crop1="-crop ${w4_p1}x${hFor43Ratio}+$xlo+$h4"
    crop2="-crop ${w4_p2}x${hFor43Ratio}+0+$h4"
    echo convert \\\( "$jpgfile" "$crop1" \\\) \\\( "$jpgfile" "$crop2" \\\) +append "${stem}_x$imgx.jpg" >> "$outshfile"
    imgid2="${imgid}_x$imgx"
    echo "UPDATE image SET usable=true WHERE position('$imgid2.jpg' in system_path) > 0;" | tee -a $sqlfile
  elif [ $imgx -lt $w8 ]; then
    # must paste together the two sides of the image
    w4_p1=$(($w8 - $imgx))
    xhi=$(($imgw - $w4_p1))
    w4_p2=$(($w4 - $w4_p1))
    crop1="-crop ${w4_p1}x${hFor43Ratio}+$xhi+$h4"
    crop2="-crop ${w4_p2}x${hFor43Ratio}+0+$h4"
    echo convert \\\( "$jpgfile" "$crop1" \\\) \\\( "$jpgfile" "$crop2" \\\) +append "${stem}_x$imgx.jpg" >> "$outshfile"
    imgid2="${imgid}_x$imgx"
    echo "UPDATE image SET usable=true WHERE position('$imgid2.jpg' in system_path) > 0;" | tee -a $sqlfile
  else
    # straightforward crop
    xlo=$(($imgx - $w8))
    crop="-crop ${w4}x${hFor43Ratio}+$xlo+$h4"
    echo convert "$jpgfile" "$crop" "${stem}_x$imgx.jpg" >> "$outshfile"
    imgid2="${imgid}_x$imgx"
    echo "UPDATE image SET usable=true WHERE position('$imgid2.jpg' in system_path) > 0;" | tee -a $sqlfile
  fi
done
