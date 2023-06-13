#!/bin/bash

HDR_threshold=0.35
HDR_rejectdir=HDR_rejects
Skimage_contrast_threshold=0.35
HDR_floor=0.8
Skimage_contrast_rejectdir=Skimage_rejects

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
matw4=$(($matw / 4))
matw8=$(($matw / 8))
xs=`grep 'Found road centres' "$outfile" | sed -e 's/.*\[\([0-9 ]*\)\]./\1/'`

ispano=

if [ ! -n "`grep 'panoramic input' "$outfile"`" ]; then
  echo Stem: $stem, Image ID: $imgid
  xboundlo=$matw4
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
  hdr="`grep 'HDR:' "$outfile" | cut -f2 -d' '`"
  hdr_awk="{ print (\$1 > \$2) }"
  hdrtest=`echo "$hdr $HDR_threshold" | awk -- "$hdr_awk"`
  echo "HDR: $hdr"
  skcon="`grep 'Skimage contrast:' "$outfile" | cut -f3 -d' '`"
  # Account for low contrast images with a reasonably high level of HDR
  # $skcon + max(0, $hdr - HDR_floor) < Skimage_contrast_threshold?
  contrast_awk="{ print(\$1 + (\$2 - \$4 > 0 ? \$2 - \$4 : 0) > \$3) }"
  skcontest=`echo "$skcon $hdr $Skimage_contrast_threshold $HDR_floor" | awk -- "$contrast_awk"`
  echo "Skimage contrast: $skcon"
  if (( $hdrtest == 0 )); then
    mkdir -p $HDR_rejectdir
    cp $outfile $jpgfile $npzfile $HDR_rejectdir
  elif (( $skcontest == 0 )); then
    mkdir -p $Skimage_contrast_rejectdir
    cp $outfile $jpgfile $npzfile $Skimage_contrast_rejectdir
  elif (( $match != 0 && $hdrtest == 1 && $skcontest == 1 )); then
    echo Acceptable image quality and road centre: $match
    echo "UPDATE image SET enabled=true WHERE position('$imgid.jpg' in system_path) > 0;" | tee $sqlfile
  else
    echo "UPDATE image SET enabled=false WHERE position('$imgid.jpg' in system_path) > 0;" | tee $sqlfile
  fi
  exit 0
fi

cat > "$outshfile" << 'EOF'
#!/bin/bash
EOF

echo -n | tee $sqlfile

xs2=""
for x in $xs; do
  xleft=$(($x - $matw8 + 10))
  xright=$(($x + $matw8 - 10))
  echo $x $xleft $xright
  if [ $xleft -lt 0 ]; then
    xleft=$(($xleft + $matw))
  elif [ $xright -ge $matw ]; then
    xright=$(($xright - $matw))
  fi
  xs2="$xs2 $xleft $x $xright"
done

xcompleted=""

for x in $xs2; do
  imgx=$(($imgw * $x / $matw))
  outjpgfile="${stem}_x$imgx.jpg"
  testprefix="[ -e '$outjpgfile' ] || "

  match=0
  wrapx=$(($x - $matw))
  # ugly but effective test for duplicates due to wrap
  for y in $xs2; do
    [ $wrapx == $y ] && match=1
  done
  # find straight-up duplicates
  for y in $xcompleted; do
    [ $x == $y ] && match=1
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
    echo "$testprefix" convert "$jpgfile" "$crop" "$outjpgfile" >> "$outshfile"
    imgid2="${imgid}_x$imgx"
    echo "UPDATE image SET enabled=true WHERE position('$imgid2.jpg' in system_path) > 0;" | tee -a $sqlfile
  elif [ $imgx -gt $xwrapneeded ]; then
    # must paste together the two sides of the image
    xlo=$(($imgx - $w8))
    w4_p1=$(($imgw - $xlo))
    w4_p2=$(($w4 - $w4_p1))
    crop1="-crop ${w4_p1}x${hFor43Ratio}+$xlo+$h4"
    crop2="-crop ${w4_p2}x${hFor43Ratio}+0+$h4"
    echo "$testprefix" convert \\\( "$jpgfile" "$crop1" \\\) \\\( "$jpgfile" "$crop2" \\\) +append "$outjpgfile" >> "$outshfile"
    imgid2="${imgid}_x$imgx"
    echo "UPDATE image SET enabled=true WHERE position('$imgid2.jpg' in system_path) > 0;" | tee -a $sqlfile
  elif [ $imgx -lt $w8 ]; then
    # must paste together the two sides of the image
    w4_p1=$(($w8 - $imgx))
    xhi=$(($imgw - $w4_p1))
    w4_p2=$(($w4 - $w4_p1))
    crop1="-crop ${w4_p1}x${hFor43Ratio}+$xhi+$h4"
    crop2="-crop ${w4_p2}x${hFor43Ratio}+0+$h4"
    echo "$testprefix" convert \\\( "$jpgfile" "$crop1" \\\) \\\( "$jpgfile" "$crop2" \\\) +append "$outjpgfile" >> "$outshfile"
    imgid2="${imgid}_x$imgx"
    echo "UPDATE image SET enabled=true WHERE position('$imgid2.jpg' in system_path) > 0;" | tee -a $sqlfile
  else
    # straightforward crop
    xlo=$(($imgx - $w8))
    crop="-crop ${w4}x${hFor43Ratio}+$xlo+$h4"
    echo "$testprefix" convert "$jpgfile" "$crop" "$outjpgfile" >> "$outshfile"
    imgid2="${imgid}_x$imgx"
    echo "UPDATE image SET enabled=true WHERE position('$imgid2.jpg' in system_path) > 0;" | tee -a $sqlfile
  fi
  xcompleted="$xcompleted $x"
done
