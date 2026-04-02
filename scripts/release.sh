

##### macOS
# zip data...
# zip -r -X esv1p1_smokeset.zip esv1p1_smoke -x "*.DS_Store"

# create hash file...
# shasum -a 256 *.zip *.md > esv1p1_256SUMS.txt

# check has file...
# shasum -a 256 -c esv1p1_256SUMS.txt



##### Windows Powershell
# zip data...


# create hash file...
# Get-ChildItem -File | Where-Object { $_.Name -ne 'esv1p1_256SUMS.txt' } | ForEach-Object { '{0}  {1}' -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash, $_.Name } | Set-Content esv1p1_256SUMS.txt

# check hash file...
# Get-Content .\esv1p1_256SUMS.txt | ForEach-Object { $p = $_ -split '  ',2; if (Test-Path $p[1]) { if ((Get-FileHash $p[1] -Algorithm SHA256).Hash -eq $p[0]) { "OK  $($p[1])" } else { "FAIL  $($p[1])" } } else { "MISSING  $($p[1])" } }