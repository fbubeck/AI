?($	뮘|???5|???J+???!H?z?@$	o?:[??@&└s?@??E???!?H*?`N@@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$J+???<Nё\???AL7?A`???Y??H.?!??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?sF????=?U?????A6?>W[???Y	?c???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&sh??|???Ș?????A?lV}????Y??q????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&7?A`?????q?????A?Zd;???Yf??a????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&&䃞ͪ???m4??@??A????Y=?U?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&HP?s???$(~??k??A?5?;N???Y?=yX???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&!?lV}????ZӼ???A??? ?r??YNё\?C??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&+??????W?/?'??Ad]?Fx??Y?4?8EG??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&H?z?@???B??@A=?U????Y???߾??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?p=
ף???? ???A?z?G???Y??JY?8??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
[Ӽ?@NbX9???A????B-@Y???{????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?x?&1??p_?Q??A??????Y??j+????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&B`??"???e?`TR'??A?[ A???Ya??+e??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?l????????+e???A?HP???Ye?`TR'??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??????????HP??A?镲q??Y??镲??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&}?5^?I????1??%??A??/?$??Y?o_???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?!??u????[ A???A??k	????Y?Q?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?V?/?'???????A??j+????Y? ?	???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&B>?٬????e??a???A?^)????Y???Q???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&"??u????{?/L?
??A?5?;N???Y??ܥ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?e?c]???s??A??A??C?l??Y[B>?٬??*	fffff??@2F
Iterator::Model1?Zd??!??y?ͶK@)??K7?A??1kJH?F@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???~?:??!%?[V??6@)9??v????1? &TU5@:Preprocessing2U
Iterator::Model::ParallelMapV2%??C???!?rY_#@)%??C???1?rY_#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipQk?w????!T?!2IF@)`??"????1?&?(5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatew??/???!k?f?D?#@)??Q???1#??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??S㥛??!???5yn@)??S㥛??1???5yn@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapd;?O????!??͂w?,@)??~j?t??1z?ͶeX@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*ݵ?|г??!ĸX#0<??)ݵ?|г??1ĸX#0<??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 56.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9m???X?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	5?? v???????f???<Nё\???!???B??@	!       "	!       *	!       2$	??Y?L????bƽ}??L7?A`???!????B-@:	!       B	!       J$	?t/?????S4)K/??	?c???!f??a????R	!       Z$	?t/?????S4)K/??	?c???!f??a????JCPU_ONLYYm???X?@b Y      Y@qڣh5?>@"?
both?Your program is POTENTIALLY input-bound because 56.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?30.8016% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 