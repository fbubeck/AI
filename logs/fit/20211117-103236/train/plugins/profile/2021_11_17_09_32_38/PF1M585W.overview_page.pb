?($	5$?AQr??S?c?+0????a??4??!6?;Nё??$	l????@?ӷ?m@7ɀ*?@!5W,@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???{????-????ƻ?A?|?5^???Y2U0*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?\?C??????&???AԚ?????Yı.n???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?W?2????j+????A
ףp=
??Yz6?>W[??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?x?&1??S??:??A?/?$??Y???H??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&+??ݓ??? c?ZB>??An????YM?J???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?c?ZB????e??a??A??{??P??Y??&???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&d]?Fx???~?:p???A??Pk?w??Y9??v????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&:??H???#??~j???A??N@a??Y?o_???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&}??b???I.?!????A??N@a??Y??6?[??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	vOjM??Tt$?????A?=yX???Y?ݓ??Z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
L?
F%u??S??:??Ad]?Fx??Yh??|?5??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?H?}???1??%???A?-?????Y?l??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ׁsF???vq?-??Axz?,C??Y?<,Ԛ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?w??#???*??D???A?I+???Y??ǘ????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ͪ?????k	??g??Ac?=yX??Y???Q???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?	?c????ͪ????AH?z?G??Y?v??/??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&6?;Nё???G?z??A??j+????YHP?s??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?c]?F???s?????Ao?ŏ1??Y?&1???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ݓ??Z???? ???A?X????Y?0?*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???Mb???+e?X??A?:pΈ???Y'???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??a??4???ǘ?????A??	h"??Y5?8EGr??*	gffff??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat{?/L?
??!??q^A@)?q??????1z??]?@:Preprocessing2F
Iterator::ModelS?!?uq??!(@?Q?A@)?W?2??1Ⱥmr?65@:Preprocessing2U
Iterator::Model::ParallelMapV2!?lV}??!?}??s-@)!?lV}??1?}??s-@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipL?
F%u??!??t;?P@)6<?R?!??1??'???#@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate??ܵ?|??!Uǂ#?0@)>yX?5ͻ?1??m?e4"@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice@a??+??!?9/%??@)@a??+??1?9/%??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?lV}???!?V?Y,4@)??@??Ǩ?1?=f?:@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??(????!?%??@)??(????1?%??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 52.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9N0o?b?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	]ʥ\ʥ??'@?.????-????ƻ?!??&???	!       "	!       *	!       2$	??e:?????{?ך????	h"??!??j+????:	!       B	!       J$	Ex~?@??????T???v??/??!??6?[??R	!       Z$	Ex~?@??????T???v??/??!??6?[??JCPU_ONLYYN0o?b?@b Y      Y@q??+o??S@"?
both?Your program is POTENTIALLY input-bound because 52.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?78.0144% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 