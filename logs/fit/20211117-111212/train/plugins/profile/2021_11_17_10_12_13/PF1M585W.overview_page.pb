?($	4ހ????8?ɯ&Q?????1????!,Ԛ????$	???@?wh?@?!??NC@!zɰb?0@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???B?i?????B?i??A?ZB>????Y???K7???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&v?????????????A^?I+??Y
ףp=
??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&#??~j????ŏ1w??A=?U????Y㥛? ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&E???JY???=yX???A?uq???Y?&S???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??+e????Pk?w???ATt$?????Y??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&,Ԛ?????1??%???A?5^?I??Y??ܥ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&6<?R?!?????(\???A?&S???Y???JY???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???N@????&???A&S??:??Y??ׁsF??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??g??s??6<?R?!??A??ݓ????Y/n????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	i o???????_vO??A?????K??Y?W[?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
ı.n???yX?5?;??AΪ??V???YV-???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&\???(\??Ӽ????A؁sF????Y? ?	???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?9#J{???d?]K???A?Fx$??YV-???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???&S??'?W???A?G?z??Y??Pk?w??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?^)??? ?~?:p??A?J?4??Y?
F%u??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&L7?A`???1?*????A?v??/??Y'???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??????????H??A???????Ya2U0*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?A`??"??h"lxz???Ae?`TR'??YZd;?O???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?[ A?c??????????AY?? ???Y?5?;Nё?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??+e???_)?Ǻ??A?x?&1??Y2U0*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???1????r??????A2??%䃾?YX9??v???*?????C?@)      @=2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat_?L???!5?wL?A@)?(\?????1?D?rlc@@:Preprocessing2F
Iterator::Model?'????!?9T,hA@)?L?J???1?i???54@:Preprocessing2U
Iterator::Model::ParallelMapV2??Ƽ?!P????+@)??Ƽ?1P????+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipΈ?????!/????xP@)V-?????1?_??!'@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice??H.?!??!?u?*?&@)??H.?!??1?u?*?&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate$(~????!????T?*@)p_?Q??1Dx?n@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?	???!p|???z2@)?/?'??1?z??}U@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*a??+e??!????@)a??+e??1????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 51.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???>q?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?g?@?8??8??>j<?????B?i??!?1??%???	!       "	!       *	!       2$	?Gǫ??W$?*??2??%䃾?!?5^?I??:	!       B	!       J$	?7?$ǆ?????????V-???!???K7???R	!       Z$	?7?$ǆ?????????V-???!???K7???JCPU_ONLYY???>q?@b Y      Y@q??dp7U@"?
both?Your program is POTENTIALLY input-bound because 51.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?84.8662% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 