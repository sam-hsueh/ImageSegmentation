﻿#pragma checksum "..\..\..\Home.xaml" "{ff1816ec-aa5e-4d10-87f7-6f4963833460}" "23BF33C9C6023FB6C4DB99BA013AB794FEA14E88"
//------------------------------------------------------------------------------
// <auto-generated>
//     此代码由工具生成。
//     运行时版本:4.0.30319.42000
//
//     对此文件的更改可能会导致不正确的行为，并且如果
//     重新生成代码，这些更改将会丢失。
// </auto-generated>
//------------------------------------------------------------------------------

using GDIWpfControl;
using MaterialDesignThemes.Wpf;
using System;
using System.Diagnostics;
using System.Windows;
using System.Windows.Automation;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Controls.Ribbon;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Forms.Integration;
using System.Windows.Ink;
using System.Windows.Input;
using System.Windows.Markup;
using System.Windows.Media;
using System.Windows.Media.Animation;
using System.Windows.Media.Effects;
using System.Windows.Media.Imaging;
using System.Windows.Media.Media3D;
using System.Windows.Media.TextFormatting;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Shell;
using WeldFeatureDetection;
using WeldFeatureDetection.Domain;


namespace WeldFeatureDetection {
    
    
    /// <summary>
    /// Home
    /// </summary>
    public partial class Home : System.Windows.Controls.UserControl, System.Windows.Markup.IComponentConnector {
        
        
        #line 47 "..\..\..\Home.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.DataGrid FileListGrid;
        
        #line default
        #line hidden
        
        
        #line 73 "..\..\..\Home.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.DataGrid FeaturesList;
        
        #line default
        #line hidden
        
        
        #line 133 "..\..\..\Home.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.RichTextBox richtextBox1;
        
        #line default
        #line hidden
        
        
        #line 159 "..\..\..\Home.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.DockPanel RectPanel;
        
        #line default
        #line hidden
        
        
        #line 160 "..\..\..\Home.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal MaterialDesignThemes.Wpf.NumericUpDown RWidth;
        
        #line default
        #line hidden
        
        
        #line 161 "..\..\..\Home.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal MaterialDesignThemes.Wpf.NumericUpDown RHeight;
        
        #line default
        #line hidden
        
        
        #line 164 "..\..\..\Home.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.RadioButton BRect;
        
        #line default
        #line hidden
        
        
        #line 165 "..\..\..\Home.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.RadioButton BPolygon;
        
        #line default
        #line hidden
        
        
        #line 167 "..\..\..\Home.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.Button MNext;
        
        #line default
        #line hidden
        
        
        #line 169 "..\..\..\Home.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.Slider Brightness;
        
        #line default
        #line hidden
        
        
        #line 170 "..\..\..\Home.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.Slider Contrast;
        
        #line default
        #line hidden
        
        
        #line 178 "..\..\..\Home.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal GDIWpfControl.GDIControlWPF GWpf;
        
        #line default
        #line hidden
        
        private bool _contentLoaded;
        
        /// <summary>
        /// InitializeComponent
        /// </summary>
        [System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [System.CodeDom.Compiler.GeneratedCodeAttribute("PresentationBuildTasks", "9.0.5.0")]
        public void InitializeComponent() {
            if (_contentLoaded) {
                return;
            }
            _contentLoaded = true;
            System.Uri resourceLocater = new System.Uri("/WeldFeatureDetection;component/home.xaml", System.UriKind.Relative);
            
            #line 1 "..\..\..\Home.xaml"
            System.Windows.Application.LoadComponent(this, resourceLocater);
            
            #line default
            #line hidden
        }
        
        [System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [System.CodeDom.Compiler.GeneratedCodeAttribute("PresentationBuildTasks", "9.0.5.0")]
        [System.ComponentModel.EditorBrowsableAttribute(System.ComponentModel.EditorBrowsableState.Never)]
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Design", "CA1033:InterfaceMethodsShouldBeCallableByChildTypes")]
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Maintainability", "CA1502:AvoidExcessiveComplexity")]
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1800:DoNotCastUnnecessarily")]
        void System.Windows.Markup.IComponentConnector.Connect(int connectionId, object target) {
            switch (connectionId)
            {
            case 1:
            
            #line 46 "..\..\..\Home.xaml"
            ((System.Windows.Controls.Button)(target)).Click += new System.Windows.RoutedEventHandler(this.Button_Click);
            
            #line default
            #line hidden
            return;
            case 2:
            this.FileListGrid = ((System.Windows.Controls.DataGrid)(target));
            
            #line 48 "..\..\..\Home.xaml"
            this.FileListGrid.SelectionChanged += new System.Windows.Controls.SelectionChangedEventHandler(this.DataGrid_SelectionChanged);
            
            #line default
            #line hidden
            return;
            case 3:
            this.FeaturesList = ((System.Windows.Controls.DataGrid)(target));
            
            #line 73 "..\..\..\Home.xaml"
            this.FeaturesList.SelectionChanged += new System.Windows.Controls.SelectionChangedEventHandler(this.FeaturesList_SelectedIndexChanged);
            
            #line default
            #line hidden
            return;
            case 4:
            this.richtextBox1 = ((System.Windows.Controls.RichTextBox)(target));
            return;
            case 5:
            this.RectPanel = ((System.Windows.Controls.DockPanel)(target));
            return;
            case 6:
            this.RWidth = ((MaterialDesignThemes.Wpf.NumericUpDown)(target));
            
            #line 160 "..\..\..\Home.xaml"
            this.RWidth.ValueChanged += new System.Windows.RoutedPropertyChangedEventHandler<System.Nullable<double>>(this.RWidth_ValueChanged);
            
            #line default
            #line hidden
            return;
            case 7:
            this.RHeight = ((MaterialDesignThemes.Wpf.NumericUpDown)(target));
            
            #line 161 "..\..\..\Home.xaml"
            this.RHeight.ValueChanged += new System.Windows.RoutedPropertyChangedEventHandler<System.Nullable<double>>(this.RWidth_ValueChanged);
            
            #line default
            #line hidden
            return;
            case 8:
            this.BRect = ((System.Windows.Controls.RadioButton)(target));
            
            #line 164 "..\..\..\Home.xaml"
            this.BRect.Checked += new System.Windows.RoutedEventHandler(this.BRect_Checked);
            
            #line default
            #line hidden
            
            #line 164 "..\..\..\Home.xaml"
            this.BRect.Unchecked += new System.Windows.RoutedEventHandler(this.BRect_Checked);
            
            #line default
            #line hidden
            return;
            case 9:
            this.BPolygon = ((System.Windows.Controls.RadioButton)(target));
            return;
            case 10:
            this.MNext = ((System.Windows.Controls.Button)(target));
            
            #line 167 "..\..\..\Home.xaml"
            this.MNext.Click += new System.Windows.RoutedEventHandler(this.MNext_Click);
            
            #line default
            #line hidden
            return;
            case 11:
            this.Brightness = ((System.Windows.Controls.Slider)(target));
            
            #line 169 "..\..\..\Home.xaml"
            this.Brightness.ValueChanged += new System.Windows.RoutedPropertyChangedEventHandler<double>(this.Contrast_ValueChanged);
            
            #line default
            #line hidden
            return;
            case 12:
            this.Contrast = ((System.Windows.Controls.Slider)(target));
            
            #line 170 "..\..\..\Home.xaml"
            this.Contrast.ValueChanged += new System.Windows.RoutedPropertyChangedEventHandler<double>(this.Contrast_ValueChanged);
            
            #line default
            #line hidden
            return;
            case 13:
            this.GWpf = ((GDIWpfControl.GDIControlWPF)(target));
            
            #line 178 "..\..\..\Home.xaml"
            this.GWpf.PreviewKeyDown += new System.Windows.Input.KeyEventHandler(this.Form_KeyDown);
            
            #line default
            #line hidden
            
            #line 178 "..\..\..\Home.xaml"
            this.GWpf.GdiContextDraw += new GDIWpfControl.GDIControlWPF.ContextMS(this.GWpf_GdiContextDraw);
            
            #line default
            #line hidden
            
            #line 178 "..\..\..\Home.xaml"
            this.GWpf.GdiMouseDownChanged += new System.EventHandler<System.Windows.Forms.MouseEventArgs>(this.GWpf_MouseDown);
            
            #line default
            #line hidden
            
            #line 178 "..\..\..\Home.xaml"
            this.GWpf.GdiMouseMoveChanged += new System.EventHandler<System.Windows.Forms.MouseEventArgs>(this.GWpf_MouseMove);
            
            #line default
            #line hidden
            
            #line 178 "..\..\..\Home.xaml"
            this.GWpf.GdiMouseUpChanged += new System.EventHandler<System.Windows.Forms.MouseEventArgs>(this.GWpf_MouseUp);
            
            #line default
            #line hidden
            
            #line 178 "..\..\..\Home.xaml"
            this.GWpf.SizeChanged += new System.Windows.SizeChangedEventHandler(this.GWpf_SizeChanged);
            
            #line default
            #line hidden
            return;
            }
            this._contentLoaded = true;
        }
    }
}

