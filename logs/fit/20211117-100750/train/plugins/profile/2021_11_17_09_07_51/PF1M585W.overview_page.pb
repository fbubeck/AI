?($	5???M???????2??jM??St??!?V?/?'??$	?y,?HE@?!#oG)@*`3?????!????,@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???<,?????9#J{??A!?lV}??Y?g??s???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?V?/?'??㥛? ???AL?
F%u??Y???&??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&}??b???HP?s???A???H.??Y???<,Ԛ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???h o????ʡE??A?HP???Yg??j+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?O??n???L?J???A??&???Y?(??0??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&6<?R?!?????JY???A'???????Y?v??/??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?G?z????:pΈ??A?:pΈ???YǺ????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&7?[ A??)??0???A?s????Y????o??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&>yX?5???:??H???A6?>W[???YX?5?;N??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	鷯????jM????A????<,??Y'???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?46<??T㥛? ??A?? ???Y??_?L??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??d?`T???"??~j??A????H??Y"??u????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&f?c]?F??X?2ı.??A??????YQ?|a2??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&V-????|a2U0??AM?O????Y/n????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&[Ӽ????W?2??A_)?Ǻ??Y"??u????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&u?????]K?=??A?? ?	??YK?=?U??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?	?c??,Ԛ????A????z??Ylxz?,C??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???????@??ǘ??A?z6?>??Y???Q???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&x??#????/n????AF%u???Y^K?=???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&=?U?????/?'??A;M?O??Yŏ1w-!??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&jM??St??	??g????A??ݓ????Ye?X???*	??????@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatRI??&???!??>Q=HB@)?!??u???1sCH?r@@:Preprocessing2F
Iterator::Model(~??k	??!???ײ?@@)?>W[????1?͙??84@:Preprocessing2U
Iterator::Model::ParallelMapV2J{?/L???!?OT?)@)J{?/L???1?OT?)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??H.?!??!?/???P@)?g??s???1?^(%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceC??6??!鵬6!c@)C??6??1鵬6!c@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::ConcatenateΪ??V???!JS??/@)??????1F??[??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap)\???(??!?I????3@)	?^)ˠ?1$?^@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*%u???!?c)?`W@)%u???1?c)?`W@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 57.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?[#?T?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	l#?oy?????,??	??g????!㥛? ???	!       "	!       *	!       2$	????:???7aP?*????ݓ????!L?
F%u??:	!       B	!       J$	f??~?????3n@ԋ?lxz?,C??!???&??R	!       Z$	f??~?????3n@ԋ?lxz?,C??!???&??JCPU_ONLYY?[#?T?@b Y      Y@q9񴂃??@"?
both?Your program is POTENTIALLY input-bound because 57.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?31.8497% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 