?($	#?^]i?????o?0????i?q????!F%u???$	mS?A?@E??K?@"??	@!?_?	?3@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?W?2??=,Ԛ???A?/?'??Y/?$???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?(??0??????B???A???<,???Y?X?? ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??K7?A????H?}??A?J?4??Y?ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&z?):?????0?*??AP??n???YX9??v???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&7?A`???????3???AQ?|a??Y??D????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???ZӼ???Y??ڊ??A?=?U???Y]m???{??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&)\???(??F%u???AB?f??j??Y?<,Ԛ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??+e????镲q??A????x???YvOjM??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&]?C?????HP?s???A?n?????Y?l??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	TR'??????X?? ??A?]K?=??Y?o_???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?z?G???0L?
F%??A??T?????Y? ?	???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&HP?s???Qk?w????A?z6?>??Yw-!?l??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&F%u???L7?A`???A_?Q???Y?0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??^??]m???{??A6<?R?!??Y?St$????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?\m??????H.?!???Am???????YM?O???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&&S????Y?8??m??Ap_?Q??Y333333??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?&1???yX?5?;??Aŏ1w-!??Y?,C????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&C?i?q???&S??:??AV}??b??YHP?sע?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&x$(~???????9#??Au????Y?{??Pk??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??_?L???JY?8???A/n????Y'???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?i?q?????7??d???A?QI??&??Y?(??0??*	43333?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?N@a???!?Tc??JC@)???1????1????HB@:Preprocessing2F
Iterator::Model䃞ͪ???!sW?/B@)???????1?.e???7@:Preprocessing2U
Iterator::Model::ParallelMapV2?Q???!2o??8m(@)?Q???12o??8m(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??K7?A??!茨U??O@)??z6???1??????"@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate)\???(??!{???$}(@)4??7?´?1??(98@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlicejM????!ڇ*??@)jM????1ڇ*??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??y?)??!?_???/@)?c]?F??1???D?+@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?p=
ף??!v?6@)?p=
ף??1v?6@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 50.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9\??+?*@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??&5????4o???=,Ԛ???!L7?A`???	!       "	!       *	!       2$	??0???\*?EX???QI??&??!_?Q???:	!       B	!       J$	]?,s?t???8-?|????ZӼ???!/?$???R	!       Z$	]?,s?t???8-?|????ZӼ???!/?$???JCPU_ONLYY\??+?*@b Y      Y@q2?L 9@"?
both?Your program is POTENTIALLY input-bound because 50.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?25.1262% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 