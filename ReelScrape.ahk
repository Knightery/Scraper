#Requires AutoHotKey v2.0
F1::
{
number := 0
loop
{
loop 250
{
	number := Random(0.56, 1.2)
	WinActivate ("ahk_exe chrome.exe")
	MouseClick "left", 515, 786
	Sleep 50
	Sleep 100
	Send "{Down}"
    Sleep(number) ; Wait for the page to load
    Send("^l") ; Focus on the address bar
    Sleep(500)
    Send("^c") ; Copy the URL
    Sleep(500)
    URL := A_Clipboard
    ; Append URL to Excel file
    xl := ComObject("Excel.Application")
    xl.Visible := True
    wb := xl.Workbooks.Open("C:\Users\nyter\Desktop\Scraper\Reels\ReelData.xlsx")
    ws := wb.Sheets(1)
    lastRow := ws.Cells(ws.Rows.Count, "A").End(-4162).Row + 1 ; Find the last row in column A
    ws.Cells(lastRow, 1).Value := URL
    wb.Save()
    wb.Close()
	xl.Quit()
}
Send "^r"
Sleep 5000
}
}
F2::ExitApp