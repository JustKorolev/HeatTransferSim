Option Explicit

' SolidWorks macro: export active assembly part-to-material assignments.
'
' Output workbook schema:
'   Part Name | Material Name
'
' The octree builder uses this workbook only as a material-name lookup. Contact
' detection and contact classification are intentionally left to Python.

Private Const SW_DOC_PART As Long = 1
Private Const SW_DOC_ASSEMBLY As Long = 2
Private Const XL_WORKBOOK_DEFAULT As Long = 51
Private Const XL_TO_LEFT As Long = -4159
Private Const UNKNOWN_MATERIAL As String = "unknown material"

Public Sub main()
    Dim swApp As Object
    Dim swModel As Object
    Dim swAssy As Object
    Dim components As Variant
    Dim outputPath As String

    On Error GoTo FatalError

    Set swApp = Application.SldWorks
    Set swModel = swApp.ActiveDoc

    If swModel Is Nothing Then
        MsgBox "Open a SolidWorks assembly before running this macro.", vbExclamation
        Exit Sub
    End If

    If swModel.GetType <> SW_DOC_ASSEMBLY Then
        MsgBox "Active document is not an assembly.", vbExclamation
        Exit Sub
    End If

    outputPath = PromptForOutputPath(swModel)
    If Len(outputPath) = 0 Then
        Exit Sub
    End If

    Set swAssy = swModel
    On Error Resume Next
    swAssy.ResolveAllLightWeightComponents True
    On Error GoTo FatalError

    components = swAssy.GetComponents(False)
    If IsEmpty(components) Then
        MsgBox "No components were found in the active assembly.", vbExclamation
        Exit Sub
    End If

    ExportComponentsToWorkbook components, outputPath
    MsgBox "Material export complete:" & vbCrLf & outputPath, vbInformation
    Exit Sub

FatalError:
    MsgBox "Material export failed: " & Err.Description, vbCritical
End Sub

Private Function PromptForOutputPath(ByVal swModel As Object) As String
    Dim defaultFolder As String
    Dim defaultName As String
    Dim chosenPath As String

    On Error Resume Next
    defaultFolder = swModel.GetPathName
    On Error GoTo 0

    If Len(defaultFolder) > 0 Then
        defaultFolder = Left$(defaultFolder, InStrRev(defaultFolder, "\") - 1)
    Else
        defaultFolder = Environ$("USERPROFILE") & "\Documents"
    End If

    On Error Resume Next
    defaultName = CleanFileName(swModel.GetTitle)
    On Error GoTo 0

    If LCase$(Right$(defaultName, 7)) = ".sldasm" Then
        defaultName = Left$(defaultName, Len(defaultName) - 7)
    End If
    If Len(defaultName) = 0 Then
        defaultName = "Assembly"
    End If

    chosenPath = InputBox("Enter output .xlsx path. Put this file in the mesh export directory as materials.xlsx.", "Export Assembly Materials", defaultFolder & "\materials.xlsx")

    chosenPath = Trim$(chosenPath)
    If Len(chosenPath) = 0 Then
        PromptForOutputPath = ""
    ElseIf LCase$(Right$(chosenPath, 5)) = ".xlsx" Then
        PromptForOutputPath = chosenPath
    Else
        PromptForOutputPath = chosenPath & ".xlsx"
    End If
End Function

Private Sub ExportComponentsToWorkbook(ByVal components As Variant, ByVal outputPath As String)
    Dim excelApp As Object
    Dim workbook As Object
    Dim sheet As Object
    Dim swComp As Object
    Dim componentIndex As Long
    Dim rowIndex As Long
    Dim sheetIndex As Long
    Dim errorNumber As Long
    Dim errorSource As String
    Dim errorDescription As String

    On Error GoTo ExportError

    Set excelApp = CreateObject("Excel.Application")
    excelApp.Visible = False
    excelApp.DisplayAlerts = False

    Set workbook = excelApp.Workbooks.Add
    For sheetIndex = workbook.Worksheets.Count To 2 Step -1
        workbook.Worksheets(sheetIndex).Delete
    Next sheetIndex

    Set sheet = workbook.Worksheets(1)
    sheet.Name = "Materials"
    WriteHeaders sheet

    rowIndex = 2
    For componentIndex = LBound(components) To UBound(components)
        Set swComp = components(componentIndex)
        If ShouldExportComponent(swComp) Then
            WriteComponentRow sheet, rowIndex, swComp
            rowIndex = rowIndex + 1
        End If
    Next componentIndex

    FormatSheet sheet
    workbook.SaveAs outputPath, XL_WORKBOOK_DEFAULT
    workbook.Close SaveChanges:=False
    excelApp.Quit
    Exit Sub

ExportError:
    errorNumber = Err.Number
    errorSource = Err.Source
    errorDescription = Err.Description

    On Error Resume Next
    If Not workbook Is Nothing Then workbook.Close SaveChanges:=False
    If Not excelApp Is Nothing Then excelApp.Quit
    On Error GoTo 0
    Err.Raise errorNumber, errorSource, errorDescription
End Sub

Private Function ShouldExportComponent(ByVal swComp As Object) As Boolean
    Dim swRefModel As Object
    Dim compPath As String
    Dim modelType As Long

    ShouldExportComponent = False
    If swComp Is Nothing Then
        Exit Function
    End If

    On Error Resume Next
    If swComp.IsSuppressed Then
        Exit Function
    End If
    Err.Clear
    compPath = swComp.GetPathName
    Err.Clear
    Set swRefModel = swComp.GetModelDoc2
    If Not swRefModel Is Nothing Then
        Err.Clear
        modelType = swRefModel.GetType
        If Err.Number = 0 Then
            ShouldExportComponent = (modelType = SW_DOC_PART)
            Exit Function
        End If
        Err.Clear
    End If
    On Error GoTo 0

    If Len(compPath) = 0 Then
        ShouldExportComponent = True
    Else
        ShouldExportComponent = (LCase$(Right$(compPath, 7)) = ".sldprt")
    End If
End Function

Private Sub WriteComponentRow(ByVal sheet As Object, ByVal rowIndex As Long, ByVal swComp As Object)
    Dim partName As String
    Dim materialName As String

    partName = SafeComponentName(swComp)
    materialName = SafeComponentMaterialName(swComp)

    sheet.Cells(rowIndex, 1).Value = partName
    sheet.Cells(rowIndex, 2).Value = materialName
End Sub

Private Function SafeComponentName(ByVal swComp As Object) As String
    Dim nameValue As String
    Dim pathValue As String

    On Error Resume Next
    nameValue = Trim$(CStr(swComp.Name2))
    If Len(nameValue) = 0 Then
        pathValue = CStr(swComp.GetPathName)
        nameValue = FileStem(pathValue)
    End If
    On Error GoTo 0

    If Len(nameValue) = 0 Then
        nameValue = "unknown component"
    End If
    SafeComponentName = nameValue
End Function

Private Function SafeComponentMaterialName(ByVal swComp As Object) As String
    Dim swPartModel As Object
    Dim configName As String
    Dim materialDb As String
    Dim materialName As String

    On Error Resume Next
    Set swPartModel = swComp.GetModelDoc2
    configName = CStr(swComp.ReferencedConfiguration)
    If Not swPartModel Is Nothing Then
        materialDb = ""
        materialName = swPartModel.GetMaterialPropertyName2(configName, materialDb)
        If Len(Trim$(materialName)) = 0 Then
            materialDb = ""
            materialName = swPartModel.GetMaterialPropertyName2("", materialDb)
        End If
    End If
    On Error GoTo 0

    materialName = Trim$(materialName)
    If Len(materialName) = 0 Then
        materialName = UNKNOWN_MATERIAL
    End If
    SafeComponentMaterialName = materialName
End Function

Private Sub WriteHeaders(ByVal sheet As Object)
    sheet.Cells(1, 1).Value = "Part Name"
    sheet.Cells(1, 2).Value = "Material Name"
End Sub

Private Sub FormatSheet(ByVal sheet As Object)
    Dim lastColumn As Long

    lastColumn = sheet.Cells(1, sheet.Columns.Count).End(XL_TO_LEFT).Column
    sheet.Range(sheet.Cells(1, 1), sheet.Cells(1, lastColumn)).Font.Bold = True
    sheet.Columns.AutoFit
    sheet.Rows(1).AutoFilter
End Sub

Private Function FileStem(ByVal pathValue As String) As String
    Dim fileName As String
    Dim dotIndex As Long

    fileName = pathValue
    If InStrRev(fileName, "\") > 0 Then
        fileName = Mid$(fileName, InStrRev(fileName, "\") + 1)
    End If
    dotIndex = InStrRev(fileName, ".")
    If dotIndex > 1 Then
        fileName = Left$(fileName, dotIndex - 1)
    End If
    FileStem = fileName
End Function

Private Function CleanFileName(ByVal value As String) As String
    Dim illegalChars As Variant
    Dim index As Long

    illegalChars = Array("\", "/", ":", "*", "?", """", "<", ">", "|")
    CleanFileName = value
    For index = LBound(illegalChars) To UBound(illegalChars)
        CleanFileName = Replace$(CleanFileName, CStr(illegalChars(index)), "_")
    Next index
End Function
